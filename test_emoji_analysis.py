import sys, json
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emoji_analysis as ea

fails = 0
def check(name, cond, extra=''):
    global fails
    print(('PASS ' if cond else 'FAIL ') + name, extra if not cond else '')
    fails += (not cond)

# --- extraction -------------------------------------------------------
check('plain', ea.extract_emojis('hi 😀 there 😡') == ['😀', '😡'])
check('skin tone single', ea.extract_emojis('👍🏽') == ['👍🏽'])
check('zwj family', ea.extract_emojis('👨‍👩‍👧 ok') == ['👨‍👩‍👧'])
check('heart vs16', ea.extract_emojis('I ❤️ you') == ['❤️'])
check('flag', ea.extract_emojis('🇮🇳🇺🇸') == ['🇮🇳', '🇺🇸'])
check('keycap', ea.extract_emojis('1️⃣ and #️⃣') == ['1️⃣', '#️⃣'])
check('digits are not emoji', ea.extract_emojis('call 123 * 4 #5') == [])
check('rainbow flag', ea.extract_emojis('🏳️‍🌈') == ['🏳️‍🌈'])
check('heart on fire', ea.extract_emojis('❤️‍🔥') == ['❤️‍🔥'])
check('lone copyright is not emoji', ea.extract_emojis('© 2026 ® ™') == [])
check('copyright with vs16 is', ea.extract_emojis('©️') == ['©️'])
check('no text', ea.extract_emojis('') == [] and ea.extract_emojis(None) == [])
check('devanagari/hindi untouched', ea.extract_emojis('नमस्ते दोस्त') == [])
check('strip', ea.strip_emojis('you 🤡 are 👨‍👩‍👧 here') == 'you are here', ea.strip_emojis('you 🤡 are 👨‍👩‍👧 here'))
check('names', ea.emoji_name('😡') == 'Pouting Face' and ea.emoji_name('🇮🇳') == 'Flag IN' and ea.emoji_name('1️⃣') == 'Keycap 1',
      (ea.emoji_name('😡'), ea.emoji_name('🇮🇳'), ea.emoji_name('1️⃣')))

# --- lexicon integrity -------------------------------------------------
import collections
check('lexicon loaded', len(ea.LEXICON) > 100, len(ea.LEXICON))
for cat, sent, w, chars, *rest in ea._SPEC:
    got = ea.extract_emojis(chars)
    # every char in the spec string must have been recognised as an emoji
    stripped = chars.replace('️', '')
    check(f'spec {cat}: all chars recognised', len(got) == len(stripped), (len(got), len(stripped)))
    for k in (rest[0] if rest else {}):
        check(f'override {k} belongs to {cat}', ea._key(k) in ea.LEXICON and ea.LEXICON[ea._key(k)][0] == cat)
allchars = [ea._key(e) for s in ea._SPEC for e in ea.extract_emojis(s[3])]
dups = [k for k, n in collections.Counter(allchars).items() if n > 1]
check('no emoji in two categories', not dups, dups)

# --- scenarios ---------------------------------------------------------
r = ea.analyze_text('')
check('empty', not r['has_emoji'] and r['risk_score'] == 0)
r = ea.analyze_text('Beautiful sunset 😍🔥👍')
check('friendly post: positive, zero risk', r['sentiment_label'] == 'Positive' and r['risk_score'] == 0 and r['total'] == 3, r)
r = ea.analyze_text("I'm dead 💀 that was so funny", toxicity=0.01)
check('"dead 💀" in harmless text stays low', r['risk_level'] == 'low', r['risk_score'])
r = ea.analyze_text('killed it at the gym today 🔪', toxicity=0.01)
check('harmless text + knife is low', r['risk_level'] == 'low', r['risk_score'])
r = ea.analyze_text('🖕')
check('emoji-only middle finger is flagged', r['risk_level'] == 'high', r['risk_score'])
r = ea.analyze_text('🤡')
check('single clown alone is only medium/low', r['risk_level'] in ('low', 'medium'), r['risk_score'])
r = ea.analyze_text('you are such a loser 🤡🤡🤡🤡', toxicity=0.85)
check('abusive text + repeated clowns = high, repeat+directed flags',
      r['risk_level'] == 'high' and any('Repeated' in f for f in r['flags']) and r['directed'], r)
r = ea.analyze_text('great job genius 😂👏', toxicity=0.6)
check('sarcasm flag when friendly emojis + toxic text', r['sarcasm_signal'], r['flags'])
r = ea.analyze_text('great job 😂👏', toxicity=0.02)
check('no sarcasm flag when text is clean', not r['sarcasm_signal'])
r = ea.analyze_text('nobody likes you 🔪💀', toxicity=0.7)
check('threat emojis with abusive text high', r['risk_level'] == 'high' and r['categories'].get('threat') == 2, r)
r = ea.analyze_text('what a day :) but also :( and -_-')
check('emoticons found', r['emoticons'] == 3 and r['total'] == 3, r['top'])
r = ea.analyze_text('see http://example.com/x or ratio :/')
check('url is not an emoticon', r['emoticons'] == 1, r['top'])
r = ea.analyze_text('🤡' * 50)
check('50 clowns capped, not absurd', r['risk_score'] < 0.8 and r['total'] == 50, r['risk_score'])
r = ea.analyze_text('mixed 😍 and 😡')
check('mixed sentiment', r['sentiment_label'] == 'Mixed', r['sentiment_label'])
r = ea.analyze_text('x 😀' * 3 + ' y 🇮🇳')
check('density in 0..1', 0 < r['density'] <= 1, r['density'])
json.dumps(r)                                   # must be serialisable
check('json ok', True)

# comment scoring: emojis never lower a score
s, e = ea.comment_score('you are trash 🤡🤡', 0.8)
check('comment score >= toxicity', s >= 0.8)
s, e = ea.comment_score('hello friend 😊', 0.01)
check('benign comment unchanged', abs(s - 0.01) < 1e-9)
s, e = ea.comment_score('🖕', 0.0)
check('emoji-only abuse in comment now crosses 0.4', s > 0.4, s)
s, e = ea.comment_score('you 🤡', 0.2)
check('single "you 🤡" does not cross 0.4', s < 0.4, s)

# --- image ------------------------------------------------------------
P, S = ea.CLIP_PRESENCE_LABELS, ea.CLIP_SENTIMENT_LABELS
img = ea.analyze_image_clip({P[0]: 90, P[1]: 10}, {S[0]: 5, S[1]: 10, S[2]: 85})
check('image hostile emojis', img['detected'] and img['sentiment_label'] == 'Hostile' and img['risk_score'] > 0.7 and img['risk_level'] == 'high', img)
img2 = ea.analyze_image_clip({P[0]: 55, P[1]: 45}, {S[0]: 0, S[1]: 0, S[2]: 100})
check('uncertain presence ignored', not img2['detected'] and img2['risk_score'] == 0 and img2['sentiment_label'] == 'None')
img3 = ea.analyze_image_clip({P[0]: 95, P[1]: 5}, {S[0]: 80, S[1]: 15, S[2]: 5})
check('friendly emoji image low risk', img3['sentiment_label'] == 'Friendly' and img3['risk_level'] == 'low')

# --- combine ---------------------------------------------------------
cap = ea.analyze_text('you 🤡🤡🤡', toxicity=0.8)
blk = ea.combine(cap, ea.analyze_text('', 0), img)
check('combine picks max', blk['overall_risk_score'] >= max(cap['risk_score'], img['risk_score'] * 0.7 / 0.9) - 1e-9, blk['overall_risk_score'])
check('combine contribution', abs(ea.combined_contribution(blk) - 0.9 * blk['overall_risk_score']) < 1e-9)
blk0 = ea.combine()
check('combine empty', not blk0['has_any'] and blk0['overall_risk_score'] == 0 and blk0['overall_sentiment_label'] == 'None')
json.dumps(blk); check('combine json ok', True)
# overall sentiment must say Mixed when a friendly caption meets a hostile picture
mix = ea.combine(ea.analyze_text('lovely 😍😍', 0.01), None, img)
check('combine: friendly caption + hostile image = Mixed', mix['overall_sentiment_label'] == 'Mixed', mix['overall_sentiment_label'])
mix2 = ea.combine(ea.analyze_text('you 🤡🤡🤡 great 😂', 0.8), ea.analyze_text('ugly 🖕 :)', 0.5), img)
check('combine: mixed components stay Mixed', mix2['overall_sentiment_label'] in ('Mixed', 'Negative'), mix2['overall_sentiment_label'])
print('\nFAILURES:', fails)
sys.exit(1 if fails else 0)
