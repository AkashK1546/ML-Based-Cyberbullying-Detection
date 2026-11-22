"""Emoji analysis for cyberbullying detection (standard library only - no extra installs).

What it does
------------
TEXT  (captions, text read from images by OCR, comments)
    * finds every emoji (incl. skin-tone, ZWJ "family/profession", flag and keycap sequences)
      and classic emoticons such as  :)  :(  >:(  -_-
    * counts them, ranks the most used ones, gives each a name and a category
      (threat / hostile / mocking / sexual / negative / positive / neutral)
    * computes an emoji SENTIMENT (-1 .. +1) and an emoji RISK score (0 .. 1)
    * the risk takes the surrounding text into account (see _context_factor) and raises flags such as
      "laughing emojis next to abusive text (possible sarcasm)" or "mocking emoji aimed at 'you'"

IMAGE (pixels of the uploaded picture)
    * CLIP zero-shot (see ml.py) tells whether the picture contains emojis / emoji stickers and whether
      they look friendly, neutral or hostile.  `analyze_image_clip` turns those probabilities into a
      risk score.

Everything returned is plain dict / list / str / float / int, so it can be stored with json.dumps().
Tune the behaviour with the constants below.
"""
import re
import unicodedata
from collections import Counter

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
RISK_HIGH = 0.4            # same cut-offs the rest of the app uses for "High" / "Medium" risk
RISK_MEDIUM = 0.2
MAX_REPEATS_COUNTED = 3    # the 4th, 5th ... copy of the same emoji adds no extra risk
REPEAT_FLAG_AT = 3         # "spam" flag when one risky emoji is used this many times
IMAGE_PRESENCE_MIN = 0.60  # CLIP must be at least this sure that emojis are in the picture
IMAGE_RISK_WEIGHT = 0.70   # CLIP zero-shot is approximate, so it counts a bit less in the combined score
EMOJI_RISK_WEIGHT = 0.90   # same weighting the app already uses for caption / OCR text scores

# Labels shared with ml.py (it feeds them to CLIP).  They are completed by the prompt
# "a social media image showing <label>" in ml._clip_probs.
CLIP_PRESENCE_LABELS = ["emojis, emoticons or emoji stickers", "no emojis and no emoticons"]
CLIP_SENTIMENT_LABELS = ["friendly, happy or loving emoji faces",
                         "neutral or plain emoji faces",
                         "angry, mocking, rude or hostile emoji faces"]


def risk_level(score):
    return 'high' if score >= RISK_HIGH else 'medium' if score >= RISK_MEDIUM else 'low'


# ---------------------------------------------------------------------------
# 1. Finding emojis in a string
# ---------------------------------------------------------------------------
ZWJ, VS16, KEYCAP = '‍', '️', '⃣'

# code points that are emoji on their own
_ALWAYS = [(0x1F000, 0x1F2FF), (0x1F300, 0x1FAFF), (0x2600, 0x27BF), (0x2B50, 0x2B50), (0x2B55, 0x2B55),
           (0x2B1B, 0x2B1C), (0x231A, 0x231B), (0x23E9, 0x23F3), (0x23F8, 0x23FA), (0x25FD, 0x25FE)]
# code points that are only an emoji when followed by the emoji variation selector U+FE0F
_NEEDS_VS16 = {0x00A9, 0x00AE, 0x203C, 0x2049, 0x2122, 0x2139, 0x2328, 0x23CF, 0x24C2, 0x25AA, 0x25AB,
               0x25B6, 0x25C0, 0x25FB, 0x25FC, 0x2934, 0x2935, 0x3030, 0x303D, 0x3297, 0x3299} \
    | set(range(0x2194, 0x219A)) | {0x21A9, 0x21AA} | set(range(0x2B05, 0x2B08))


def _is_skin_tone(cp):
    return 0x1F3FB <= cp <= 0x1F3FF


def _is_regional(cp):
    return 0x1F1E6 <= cp <= 0x1F1FF


def _is_base(cp):
    if _is_skin_tone(cp) or _is_regional(cp):
        return False
    return any(a <= cp <= b for a, b in _ALWAYS)


def _absorb_modifiers(text, j):
    """Skip variation selectors, skin tones and emoji 'tag' characters that belong to the previous emoji."""
    n = len(text)
    while j < n:
        cp = ord(text[j])
        if text[j] == VS16 or _is_skin_tone(cp) or 0xE0020 <= cp <= 0xE007F:
            j += 1
        else:
            break
    return j


def _scan(text):
    """Return [(start, end)] for every emoji in `text`."""
    spans, i, n = [], 0, len(text)
    while i < n:
        ch = text[i]
        cp = ord(ch)
        if ch in '#*0123456789':                              # keycap  1️⃣  #️⃣
            j = i + 1
            if j < n and text[j] == VS16:
                j += 1
            if j < n and text[j] == KEYCAP:
                spans.append((i, j + 1))
                i = j + 1
            else:
                i += 1
            continue
        if _is_regional(cp):                                  # flag = two regional indicators
            if i + 1 < n and _is_regional(ord(text[i + 1])):
                spans.append((i, i + 2))
                i += 2
            else:
                i += 1
            continue
        if _is_base(cp) or (cp in _NEEDS_VS16 and i + 1 < n and text[i + 1] == VS16):
            j = _absorb_modifiers(text, i + 1)
            while j + 1 < n and text[j] == ZWJ:               # 👨‍👩‍👧   🏳️‍🌈   ❤️‍🔥
                nxt = ord(text[j + 1])
                if _is_base(nxt) or nxt in _NEEDS_VS16 or _is_regional(nxt):
                    j = _absorb_modifiers(text, j + 2)
                else:
                    break
            spans.append((i, j))
            i = j
            continue
        i += 1
    return spans


def extract_emojis(text):
    """All emojis in `text`, in order (repeats kept)."""
    text = text or ''
    return [text[a:b] for a, b in _scan(text)]


def strip_emojis(text):
    """`text` with every emoji replaced by a space."""
    text = text or ''
    out, last = [], 0
    for a, b in _scan(text):
        out.append(text[last:a])
        last = b
    out.append(text[last:])
    return ' '.join(''.join(out).split())


def _key(emoji):
    """Normalised lookup key: first symbol of the sequence, without variation selector / skin tone."""
    for ch in emoji:
        cp = ord(ch)
        if ch in (VS16, ZWJ, KEYCAP) or _is_skin_tone(cp) or 0xE0020 <= cp <= 0xE007F:
            continue
        return ch
    return emoji


def emoji_name(emoji):
    if emoji.endswith(KEYCAP):
        return 'Keycap ' + emoji[0]
    cps = [c for c in emoji if _is_regional(ord(c))]
    if len(cps) == 2 and len(emoji) == 2:
        return 'Flag ' + ''.join(chr(ord(c) - 0x1F1E6 + ord('A')) for c in cps)
    try:
        name = unicodedata.name(_key(emoji)).replace('_', ' ').title()
    except ValueError:                                         # emoji newer than this Python's Unicode data
        name = 'Emoji'
    return name + (' (sequence)' if ZWJ in emoji else '')


# ---------------------------------------------------------------------------
# 2. Lexicon: category, sentiment and abuse-risk weight of the emojis that matter for bullying
# ---------------------------------------------------------------------------
# (category, sentiment -1..1, default risk weight 0..1, emojis, {emoji: own risk weight})
_SPEC = [
    ('threat', -0.8, 0.50, '🔪🗡⚔🔫💣🧨🪓🩸⚰🪦☠💀',
     {'💀': 0.30, '⚔': 0.30, '🧨': 0.30, '🩸': 0.40, '🪦': 0.40, '☠': 0.45, '💣': 0.45}),
    ('hostile', -0.8, 0.35, '😡🤬😠👿😈👹👺💢🖕👊🥊',
     {'🤬': 0.50, '👿': 0.45, '🖕': 0.70, '😈': 0.30, '👊': 0.30, '🥊': 0.20, '💢': 0.30}),
    ('mocking', -0.5, 0.30, '🤡💩🤮🤢🐷🐖🐍🐀🙄😒😏🙃🥱🤓👎🫵🗑🚮',
     {'🤡': 0.40, '💩': 0.40, '🤮': 0.35, '🐷': 0.35, '🐖': 0.35, '🙄': 0.25, '😒': 0.20, '😏': 0.15,
      '🙃': 0.15, '🥱': 0.15, '🤓': 0.20, '🫵': 0.15, '🗑': 0.25, '🚮': 0.25, '🐍': 0.20, '🐀': 0.25}),
    ('sexual', -0.2, 0.30, '🍆🍑💦👅🔞',
     {'🍆': 0.40, '🔞': 0.50, '🍑': 0.25, '👅': 0.25}),
    ('negative', -0.6, 0.0, '😢😭😞😔😟😩😫😖💔😰😨😱😓😿🥀😥😪🥺'),
    ('positive', 0.8, 0.0, '😀😃😄😁😆😊🙂😉😍🥰😘😗😙😚🤗👍👏🙌🎉🎊💪👌🤝🙏😇🥳🤩😎✨🌟⭐🔥❤🧡💛💚💙💜'
                           '💕💖💗💓💞💝😅😋😌☺😂🤣😹😻👑🏆💯🥹'),
]
LAUGHING = {'😂', '🤣', '😆', '😹', '💀', '🙃', '😏'}          # used to mock people as well as to laugh

CATEGORY_LABELS = {
    'threat': 'Threat / violence', 'hostile': 'Hostile / angry', 'mocking': 'Mocking / insulting',
    'sexual': 'Sexual / suggestive', 'negative': 'Sad / distressed', 'positive': 'Positive / friendly',
    'neutral': 'Neutral / other',
}
RISKY_CATEGORIES = ('threat', 'hostile', 'mocking', 'sexual')

LEXICON = {}                # key -> (category, sentiment, risk weight)
for _cat, _sent, _w, _chars, *_rest in _SPEC:
    _own = {_key(k): v for k, v in (_rest[0] if _rest else {}).items()}
    for _e in extract_emojis(_chars):
        LEXICON.setdefault(_key(_e), (_cat, _sent, _own.get(_key(_e), _w)))
_LAUGH_KEYS = {_key(e) for e in LAUGHING}

# classic text emoticons  (matched only as whole whitespace-separated tokens, so "http://" is safe)
_EMOTICONS = {
    ':)': 'positive', ':-)': 'positive', '(:': 'positive', ':D': 'positive', ':-D': 'positive',
    'xD': 'positive', 'XD': 'positive', ';)': 'positive', ';-)': 'positive', ':P': 'positive',
    ':p': 'positive', ':-P': 'positive', '<3': 'positive', ':*': 'positive', ':3': 'positive',
    '^_^': 'positive', '^^': 'positive',
    ':(': 'negative', ':-(': 'negative', ":'(": 'negative', ':/': 'negative', ':-/': 'negative',
    'T_T': 'negative', ':c': 'negative', '</3': 'negative', 'D:': 'negative',
    '>:(': 'hostile', '>:-(': 'hostile', ':@': 'hostile', '>_<': 'hostile', 'grr': 'hostile',
    '-_-': 'mocking', ':|': 'neutral', ':O': 'neutral', ':o': 'neutral',
}
_EMO_RISK = {'hostile': 0.30, 'mocking': 0.20}
_SECOND_PERSON = {'you', 'u', 'ur', 'your', 'youre', 'yours', 'yourself', 'ya', 'yall', 'urself', 'thou'}


def _lookup(symbol, kind):
    if kind == 'emoticon':
        cat = _EMOTICONS[symbol]
        sent = {'positive': 0.7, 'negative': -0.5, 'hostile': -0.8, 'mocking': -0.4}.get(cat, 0.0)
        return cat, sent, _EMO_RISK.get(cat, 0.0)
    return LEXICON.get(_key(symbol), ('neutral', 0.0, 0.0))


# ---------------------------------------------------------------------------
# 3. Text analysis
# ---------------------------------------------------------------------------
def _find_emoticons(text_without_emoji):
    found = []
    for tok in text_without_emoji.split():
        core = tok.rstrip('.,!?')
        for cand in (tok, core):
            if cand in _EMOTICONS:
                found.append(cand)
                break
    return found


def _words(text):
    return re.findall(r"[^\W_]+(?:'[^\W_]+)?", text.lower().replace('’', "'"))


def _context_factor(has_text, toxicity, directed):
    """How seriously a risky emoji should be taken, given the words around it.

    * abusive text  -> the emoji reinforces it            (1.0)
    * emoji only    -> nothing else to excuse it          (0.8)
    * harmless text -> "I'm dead 💀 lol", "killed it 🔪"   (0.35)
    * aimed at someone ("you 🤡") -> at least 0.9
    """
    if toxicity is not None and toxicity >= 0.3:
        c = 1.0
    elif not has_text:
        c = 0.8
    else:
        c = 0.35
    return max(c, 0.9) if directed else c


def _empty_text_result():
    return {'has_emoji': False, 'total': 0, 'unique': 0, 'emoticons': 0, 'density': 0.0,
            'top': [], 'categories': {}, 'sentiment_score': None, 'sentiment_label': 'None',
            'risk_score': 0.0, 'risk_level': 'low', 'flags': [], 'sarcasm_signal': False,
            'directed': False, 'risky_emojis': []}


def analyze_text(text, toxicity=None):
    """Emoji analysis of one piece of text.

    toxicity: the text's toxicity score (0..1) from the language model, if you have it.  It makes the
    emoji risk context-aware.  Works without it too.
    """
    text = text or ''
    res = _empty_text_result()
    plain = strip_emojis(text)
    items = [(e, 'emoji') for e in extract_emojis(text)] + [(e, 'emoticon') for e in _find_emoticons(plain)]
    if not items:
        return res

    words = _words(plain)
    directed = any(w.replace("'", '') in _SECOND_PERSON for w in words)
    ctx = _context_factor(bool(words), toxicity, directed)

    counts = Counter(items)
    cat_counts, sentiments, keep_prob = Counter(), [], 1.0
    top, risky = [], []
    for (sym, kind), n in counts.items():
        cat, sent, w = _lookup(sym, kind)
        cat_counts[cat] += n
        sentiments += [sent] * n
        top.append({'emoji': sym, 'name': emoji_name(sym) if kind == 'emoji' else 'Emoticon',
                    'count': n, 'category': cat, 'category_label': CATEGORY_LABELS.get(cat, cat),
                    'kind': kind})
        if w > 0:
            risky.append({'emoji': sym, 'count': n, 'category': cat})
            for _ in range(min(n, MAX_REPEATS_COUNTED)):
                keep_prob *= 1.0 - min(1.0, w * ctx)
    top.sort(key=lambda d: (-d['count'], d['emoji']))
    risk = round(1.0 - keep_prob, 4)

    total = sum(counts.values())
    avg = sum(sentiments) / total
    n_pos, n_neg = cat_counts['positive'], sum(cat_counts[c] for c in ('negative',) + RISKY_CATEGORIES)
    label = ('Positive' if avg >= 0.25 else 'Negative' if avg <= -0.25
             else 'Mixed' if (n_pos and n_neg) else 'Neutral')

    # ---- flags ---------------------------------------------------------
    flags, risky_syms = [], ' '.join(r['emoji'] for r in risky)
    if risky:
        flags.append(f'Hostile, mocking or threatening emojis used ({risky_syms})')
    spam = [r for r in risky if r['count'] >= REPEAT_FLAG_AT]
    if spam:
        flags.append('Repeated risky emoji (' + ', '.join(f"{r['emoji']} ×{r['count']}" for r in spam) + ')')
    if risky and directed:
        flags.append('Risky emojis aimed at a person ("you" / "your")')
    laugh_or_pos = any(_key(s) in _LAUGH_KEYS or _lookup(s, k)[0] == 'positive' for s, k in counts)
    sarcasm = bool(laugh_or_pos and toxicity is not None and toxicity >= RISK_HIGH)
    if sarcasm:
        flags.append('Friendly or laughing emojis next to abusive text (possible sarcasm / mockery)')
    if cat_counts['sexual']:
        flags.append('Sexually suggestive emojis')

    res.update({
        'has_emoji': True, 'total': total, 'unique': len(counts),
        'emoticons': sum(n for (_s, k), n in counts.items() if k == 'emoticon'),
        'density': round(total / (len(words) + total), 3),       # share of all tokens that are emojis
        'top': top[:8], 'categories': {c: n for c, n in cat_counts.most_common()},
        'sentiment_score': round(avg, 3), 'sentiment_label': label,
        'risk_score': risk, 'risk_level': risk_level(risk), 'flags': flags,
        'sarcasm_signal': sarcasm, 'directed': bool(directed and risky), 'risky_emojis': risky[:8],
    })
    return res


def comment_score(text, toxicity):
    """(final_score, emoji_result) for a comment.  Emojis can only RAISE the score, never lower it."""
    emo = analyze_text(text, toxicity)
    return max(toxicity or 0.0, EMOJI_RISK_WEIGHT * emo['risk_score']), emo


# ---------------------------------------------------------------------------
# 4. Image analysis (CLIP zero-shot probabilities, see ml.py)
# ---------------------------------------------------------------------------
def analyze_image_clip(presence, sentiment):
    """presence / sentiment: {label: percent} from ml._clip_probs (each dict sums to ~100)."""
    p_present = presence[CLIP_PRESENCE_LABELS[0]] / 100.0
    friendly, neutral, hostile = (sentiment[l] / 100.0 for l in CLIP_SENTIMENT_LABELS)
    detected = p_present >= IMAGE_PRESENCE_MIN
    risk = round(p_present * hostile, 4) if detected else 0.0
    top = max(('Friendly', friendly), ('Neutral', neutral), ('Hostile', hostile), key=lambda t: t[1])
    return {
        'detected': bool(detected),
        'presence_confidence': round(p_present * 100, 2),
        'sentiment_label': top[0] if detected else 'None',
        'sentiment_probs': {'friendly': round(friendly * 100, 2), 'neutral': round(neutral * 100, 2),
                            'hostile': round(hostile * 100, 2)},
        'sentiment_score': round(friendly - hostile, 3) if detected else None,
        'risk_score': risk, 'risk_level': risk_level(risk * IMAGE_RISK_WEIGHT / EMOJI_RISK_WEIGHT),
        'method': 'CLIP zero-shot (approximate)',
    }


def empty_image_result():
    return {'detected': False, 'presence_confidence': 0.0, 'sentiment_label': 'None',
            'sentiment_probs': {'friendly': 0.0, 'neutral': 0.0, 'hostile': 0.0}, 'sentiment_score': None,
            'risk_score': 0.0, 'risk_level': 'low', 'method': 'not available', 'available': False}


# ---------------------------------------------------------------------------
# 5. Everything together (one post)
# ---------------------------------------------------------------------------
def combine(caption=None, ocr=None, image=None):
    """Merge the caption / OCR-text / image results into the single block stored with the post."""
    caption = caption or _empty_text_result()
    ocr = ocr or _empty_text_result()
    image = image or empty_image_result()

    n_text = caption['total'] + ocr['total']
    weighted = (caption['total'] * (caption['sentiment_score'] or 0) +
                ocr['total'] * (ocr['sentiment_score'] or 0))
    parts, weight = weighted, n_text
    if image.get('detected'):
        parts += image['sentiment_score'] or 0
        weight += 1
    avg = (parts / weight) if weight else None
    # an average can hide a clash (a friendly caption on a hostile picture), so say "Mixed" when both occur
    seen = [c['sentiment_label'] for c in (caption, ocr) if c['has_emoji']]
    if image.get('detected'):
        seen.append(image['sentiment_label'])
    disagree = (any(s in ('Positive', 'Friendly') for s in seen)
                and any(s in ('Negative', 'Hostile') for s in seen))
    if avg is None:
        label = 'None'
    elif disagree:                                   # friendly and hostile sources: never hide that in an average
        label = 'Mixed'
    elif avg >= 0.25:
        label = 'Positive'
    elif avg <= -0.25:
        label = 'Negative'
    else:
        label = 'Mixed' if 'Mixed' in seen else 'Neutral'

    risk = max(caption['risk_score'], ocr['risk_score'], image['risk_score'] * IMAGE_RISK_WEIGHT / EMOJI_RISK_WEIGHT)
    risk = round(risk, 4)
    return {
        'caption': caption, 'ocr': ocr, 'image': image,
        'total_text_emojis': n_text,
        'overall_sentiment_score': None if avg is None else round(avg, 3),
        'overall_sentiment_label': label,
        'overall_risk_score': risk, 'overall_risk_level': risk_level(risk),
        'has_any': bool(n_text or image.get('detected')),
    }


def combined_contribution(block):
    """The value that goes into max(...) of the post's combined bullying score."""
    return EMOJI_RISK_WEIGHT * (block or {}).get('overall_risk_score', 0.0)
