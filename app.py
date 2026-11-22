import os
import json
import uuid
import shutil
import threading
from datetime import datetime

from flask import (abort, Flask, render_template, request, redirect, url_for, flash,
                   session, jsonify, g)
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text as sql_text, inspect as sa_inspect
from sqlalchemy.orm import selectinload
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps

import ml
import emoji_analysis

# ============================================================
# Configuration
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
POSTS_DIR = os.path.join(STATIC_DIR, 'uploads', 'posts')
PROFILES_DIR = os.path.join(STATIC_DIR, 'uploads', 'profiles')
REELS_DIR = os.path.join(STATIC_DIR, 'uploads', 'reels')
DEFAULT_AVATAR = 'default_profile.png'

# A user whose reputation drops BELOW this value is blocked from uploading,
# posting, commenting and following.
BLOCK_THRESHOLD = float(os.environ.get('BLOCK_THRESHOLD', 5.0))
START_REPUTATION = 10.0
POST_BULLYING_THRESHOLD = 0.4      # combined image/OCR/caption score; above this = bullying
COMMENT_BULLYING_THRESHOLD = 0.4   # comment score above this = bullying
MAX_POSTS_PER_PAGE = 30
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///instagram.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 60 * 60 * 24      # let browsers cache static files


def _secret_key():
    """Stable secret key: env var, else one persisted in instance/ (not hard-coded)."""
    if os.environ.get('SECRET_KEY'):
        return os.environ['SECRET_KEY']
    os.makedirs(app.instance_path, exist_ok=True)
    path = os.path.join(app.instance_path, 'secret.key')
    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write(uuid.uuid4().hex + uuid.uuid4().hex)
    with open(path) as f:
        return f.read().strip()


app.config['SECRET_KEY'] = _secret_key()
db = SQLAlchemy(app)


def get_or_404(model, ident):
    """db.get_or_404 only exists in Flask-SQLAlchemy 3+; this works on every version."""
    obj = db.session.get(model, ident)
    if obj is None:
        abort(404)
    return obj


# ============================================================
# Database models
# ============================================================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    bio = db.Column(db.Text, default='')
    profile_pic = db.Column(db.String(200), default=DEFAULT_AVATAR)
    reputation_score = db.Column(db.Float, default=START_REPUTATION)
    is_private = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    posts = db.relationship('Post', backref='author', lazy=True)
    reels = db.relationship('Reel', backref='author', lazy=True)
    comments = db.relationship('Comment', backref='author', lazy=True)
    followers = db.relationship('Follow', foreign_keys='Follow.followed_id', backref='followed', lazy='dynamic')
    following = db.relationship('Follow', foreign_keys='Follow.follower_id', backref='follower', lazy='dynamic')
    likes = db.relationship('Like', backref='user', lazy=True)
    shares = db.relationship('Share', backref='user', lazy=True)
    follow_requests_sent = db.relationship('FollowRequest', foreign_keys='FollowRequest.requester_id',
                                           backref='requester', lazy='dynamic')
    follow_requests_received = db.relationship('FollowRequest', foreign_keys='FollowRequest.target_id',
                                               backref='target', lazy='dynamic')

    @property
    def is_blocked(self):
        return self.reputation_score < BLOCK_THRESHOLD


class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image = db.Column(db.String(200), nullable=False)
    caption = db.Column(db.Text)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    is_video = db.Column(db.Boolean, default=False)
    analysis_json = db.Column(db.Text)            # cached AI analysis (computed once, at upload)

    comments = db.relationship('Comment', backref='post', lazy=True, cascade='all, delete-orphan')
    likes = db.relationship('Like', backref='post', lazy=True, cascade='all, delete-orphan')
    shares = db.relationship('Share', backref='post', lazy=True, cascade='all, delete-orphan')


class Reel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    video = db.Column(db.String(200), nullable=False)
    caption = db.Column(db.Text)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    post_id = db.Column(db.Integer, db.ForeignKey('post.id'), nullable=False, index=True)
    bullying_score = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Like(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    post_id = db.Column(db.Integer, db.ForeignKey('post.id'), nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Share(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    post_id = db.Column(db.Integer, db.ForeignKey('post.id'), nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Follow(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    follower_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    followed_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class FollowRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    requester_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    target_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    status = db.Column(db.String(20), default='pending')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# ============================================================
# Start-up helpers: folders, default avatar, legacy files, schema
# ============================================================
def ensure_folders_and_files():
    for d in (POSTS_DIR, PROFILES_DIR, REELS_DIR):
        os.makedirs(d, exist_ok=True)

    # Older versions saved everything to static/images (posts) and static/images/profiles.
    legacy = os.path.join(STATIC_DIR, 'images')
    if os.path.isdir(legacy):
        for name in os.listdir(legacy):
            src = os.path.join(legacy, name)
            if os.path.isfile(src):
                dst_dir = PROFILES_DIR if name == DEFAULT_AVATAR else POSTS_DIR
                dst = os.path.join(dst_dir, name)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
        legacy_profiles = os.path.join(legacy, 'profiles')
        if os.path.isdir(legacy_profiles):
            for name in os.listdir(legacy_profiles):
                dst = os.path.join(PROFILES_DIR, name)
                if not os.path.exists(dst):
                    shutil.move(os.path.join(legacy_profiles, name), dst)

    # The old create_directories.py wrote an EMPTY default_profile.png (a broken image).
    avatar = os.path.join(PROFILES_DIR, DEFAULT_AVATAR)
    if not os.path.exists(avatar) or os.path.getsize(avatar) == 0:
        img = Image.new('RGB', (200, 200), (200, 204, 214))
        from PIL import ImageDraw
        d = ImageDraw.Draw(img)
        d.ellipse((70, 40, 130, 100), fill=(250, 250, 252))
        d.ellipse((35, 110, 165, 240), fill=(250, 250, 252))
        img.save(avatar)


def ensure_schema():
    """Add new column / indexes to an existing database (db.create_all() never alters tables)."""
    db.create_all()
    insp = sa_inspect(db.engine)
    cols = {c['name'] for c in insp.get_columns('post')}
    with db.engine.begin() as conn:
        if 'analysis_json' not in cols:
            conn.execute(sql_text('ALTER TABLE post ADD COLUMN analysis_json TEXT'))
        for name, table, columns in (
            ('ix_post_user_id', 'post', 'user_id'), ('ix_post_created_at', 'post', 'created_at'),
            ('ix_comment_post_id', 'comment', 'post_id'), ('ix_like_post_id', 'like', 'post_id'),
            ('ix_like_user_id', 'like', 'user_id'), ('ix_follow_follower', 'follow', 'follower_id'),
            ('ix_follow_followed', 'follow', 'followed_id'),
            ('ix_share_post_id', 'share', 'post_id'),
            ('ix_followreq_target', 'follow_request', 'target_id'),
        ):
            conn.execute(sql_text(f'CREATE INDEX IF NOT EXISTS {name} ON "{table}" ({columns})'))


# ============================================================
# Image helpers
# ============================================================
def allowed_file(filename, allowed=ALLOWED_EXTENSIONS):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed


def save_image(file_storage, folder, max_side=1280, prefix=''):
    """Validate, orient, downscale and save an uploaded image. Returns the file name.

    Feed images used to be stored/served at full resolution (up to 16 MB each), which was the
    main reason the pages loaded slowly. Raises ValueError if the file is not a real image.
    """
    try:
        file_storage.stream.seek(0)
        img = Image.open(file_storage.stream)
        img.load()
    except Exception:
        raise ValueError('The file is not a valid image.')

    animated = getattr(img, 'is_animated', False)
    if animated:                                     # keep animated GIFs untouched
        name = f"{prefix}{uuid.uuid4().hex}.gif"
        file_storage.stream.seek(0)
        file_storage.save(os.path.join(folder, name))
        return name

    img = ImageOps.exif_transpose(img)
    img.thumbnail((max_side, max_side))
    if img.mode in ('RGBA', 'LA', 'P'):
        img = img.convert('RGBA')
        name, fmt, kwargs = f"{prefix}{uuid.uuid4().hex}.png", 'PNG', {'optimize': True}
    else:
        img = img.convert('RGB')
        name, fmt, kwargs = f"{prefix}{uuid.uuid4().hex}.jpg", 'JPEG', {'quality': 85, 'optimize': True}
    os.makedirs(folder, exist_ok=True)
    img.save(os.path.join(folder, name), fmt, **kwargs)
    return name


# ============================================================
# AI analysis of a post (done ONCE at upload, then cached in the DB)
# ============================================================
def analyze_post(image_path, caption):
    """CLIP (image) + EasyOCR->toxicity model (text in image) + toxicity model (caption)."""
    ml.ensure_ready()
    ocr = ml.extract_text_from_image(image_path)
    ocr_text = ocr['text'].strip()
    ocr_score = ml.toxicity_score(ocr_text) if ocr_text else 0.0

    predictions, image_score, clip_probs = ml.predict_image_full(image_path)
    caption_score = ml.toxicity_score(caption) if caption else 0.0

    # Emoji analysis: emojis in the caption + in text read from the image (text side) and
    # emojis / emoji stickers visible in the picture itself (image side, CLIP).
    emoji = emoji_analysis.combine(
        emoji_analysis.analyze_text(caption, caption_score if caption else None),
        emoji_analysis.analyze_text(ocr_text, ocr_score if ocr_text else None),
        ml.emoji_in_image(clip_probs))

    combined = max(image_score, ocr_score * 0.9, caption_score * 0.9,
                   emoji_analysis.combined_contribution(emoji))
    detected = combined > POST_BULLYING_THRESHOLD
    overall = predictions['overall']

    return {
        'emoji': emoji,
        'emoji_risk_score': float(emoji['overall_risk_score']),
        'image_bullying_score': float(image_score),
        'clip_bullying_score': float(image_score),
        'image_predictions': predictions,
        'ocr_text': ocr_text,
        'ocr_detail': {'text': ocr_text, 'word_count': ocr['word_count'],
                       'detections': ocr['detections']},
        'ocr_confidence': float(ocr['confidence']),
        'ocr_bullying_score': float(ocr_score),
        'text_bullying_score': float(caption_score),
        'combined_score': float(combined),
        'clip_overall_label': overall['predicted'],
        'clip_overall_conf': overall['confidence'] * 100,
        'bullying_detected': bool(detected),
        'overall_status': 'Cyberbullying Detected' if detected else 'Safe Content',
    }


def get_post_analysis(post):
    """Cached analysis for a post; computed (and saved) the first time for older posts."""
    path = os.path.join(POSTS_DIR, post.image)
    if post.analysis_json:
        try:
            cached = json.loads(post.analysis_json)
        except ValueError:
            cached = None
        if cached is not None:
            if 'emoji' not in cached:                    # post analysed before emoji analysis existed
                add_emoji_to_cached(post, cached, path)
            return cached
    if not os.path.exists(path):
        return None
    result = analyze_post(path, post.caption or '')
    post.analysis_json = json.dumps(result)
    db.session.commit()
    return result


def add_emoji_to_cached(post, a, path):
    """Add the emoji block to an analysis that was stored without one.

    The text part is instant. The image part needs CLIP: it is added when the models are available and
    the picture still exists; the result is only saved when it is complete. The post's stored combined
    score / verdict / reputation are NOT changed retroactively.
    """
    caption = post.caption or ''
    ocr_text = a.get('ocr_text') or ''
    image = emoji_analysis.empty_image_result()
    complete = False
    if os.path.exists(path) and ml.hub.ready.is_set():   # never block the page waiting for model loading
        try:
            _preds, _score, clip_probs = ml.predict_image_full(path)
            image = ml.emoji_in_image(clip_probs)
            complete = True
        except Exception:                                # models not loaded etc. -> show the text part only
            app.logger.info('Emoji image check skipped for post %s', post.id, exc_info=True)
    a['emoji'] = emoji_analysis.combine(
        emoji_analysis.analyze_text(caption, a.get('text_bullying_score') if caption else None),
        emoji_analysis.analyze_text(ocr_text, a.get('ocr_bullying_score') if ocr_text else None),
        image)
    a['emoji_risk_score'] = float(a['emoji']['overall_risk_score'])
    if complete:
        post.analysis_json = json.dumps(a)
        db.session.commit()


# ============================================================
# Request lifecycle: load the user once per request, enforce login
# ============================================================
PUBLIC_ENDPOINTS = {'static', 'login', 'register', 'logout'}
JSON_ENDPOINTS = {'like_post', 'share_post', 'add_comment', 'follow_user', 'handle_follow_request'}


@app.before_request
def load_user():
    g.user = None
    if request.endpoint is None or request.endpoint == 'static':
        return
    if 'user_id' in session:
        g.user = db.session.get(User, session['user_id'])
        if g.user is None:
            session.pop('user_id', None)
    if g.user is None and request.endpoint not in PUBLIC_ENDPOINTS:
        if request.endpoint in JSON_ENDPOINTS:
            return jsonify({'error': 'Not logged in'}), 401
        return redirect(url_for('login'))


def get_current_user():
    return g.get('user')


def restriction_message(user):
    return (f'Your account is restricted: your reputation score ({user.reputation_score:.1f}/10) is below '
            f'{BLOCK_THRESHOLD:g}. You cannot upload, post, comment or follow.')


def emoji_brief(emo):
    """Small summary of an emoji_analysis.analyze_text result, for showing next to a comment."""
    return {'total': emo['total'], 'sentiment': emo['sentiment_label'], 'risk_level': emo['risk_level'],
            'risk_score': emo['risk_score'], 'flags': emo['flags'],
            'top': [{'emoji': t['emoji'], 'count': t['count']} for t in emo['top'][:5]]}


@app.template_filter('emoji_brief')
def emoji_brief_filter(text, toxicity=None):
    """{{ comment.text|emoji_brief(comment.bullying_score) }} -> summary dict, or None if no emojis.

    (The stored comment score already includes the emoji risk, so this is a display-time estimate that is
    always consistent with, or slightly stricter than, the decision taken when the comment was posted.)
    """
    emo = emoji_analysis.analyze_text(text, toxicity)
    return emoji_brief(emo) if emo['has_emoji'] else None


@app.context_processor
def inject_globals():
    user = g.get('user')
    pending = 0
    if user is not None and user.is_private:
        pending = FollowRequest.query.filter_by(target_id=user.id, status='pending').count()
    return {
        'user': user,
        'pending_requests_count': pending,
        'block_threshold': BLOCK_THRESHOLD,
        'is_blocked': bool(user and user.is_blocked),
        'avatar': lambda n: url_for('static', filename='uploads/profiles/' + (n or DEFAULT_AVATAR)),
        'post_media': lambda n: url_for('static', filename='uploads/posts/' + n),
        'reel_media': lambda n: url_for('static', filename='uploads/reels/' + n),
    }


# ============================================================
# Auth
# ============================================================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if g.user:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            flash('Logged in successfully!', 'success')
            return redirect(url_for('index'))
        flash('Invalid username or password', 'error')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if g.user:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        full_name = request.form.get('full_name', '').strip()

        if not (username and email and password and full_name):
            flash('Please fill in all fields', 'error')
            return redirect(url_for('register'))
        if len(password) < 6:
            flash('Password must be at least 6 characters', 'error')
            return redirect(url_for('register'))
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return redirect(url_for('register'))
        if User.query.filter_by(email=email).first():
            flash('Email already exists', 'error')
            return redirect(url_for('register'))

        db.session.add(User(username=username, email=email, full_name=full_name,
                            password=generate_password_hash(password)))
        db.session.commit()
        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('Logged out successfully', 'success')
    return redirect(url_for('login'))


# ============================================================
# Upload (blocked users rejected; analysis stored on the post)
# ============================================================
@app.route('/upload', methods=['POST'])
def upload_post():
    user = g.user
    if user.is_blocked:
        flash(restriction_message(user), 'error')
        return redirect(url_for('index'))

    file = request.files.get('image')
    caption = request.form.get('caption', '').strip()[:500]
    if not file or not file.filename:
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload an image (PNG, JPG, JPEG, GIF, BMP, WEBP).', 'error')
        return redirect(url_for('index'))

    try:
        ml.ensure_ready()
    except ml.ModelsUnavailable as e:
        flash(str(e), 'error')
        return redirect(url_for('index'))

    try:
        filename = save_image(file, POSTS_DIR)
    except ValueError as e:
        flash(str(e), 'error')
        return redirect(url_for('index'))

    try:
        analysis = analyze_post(os.path.join(POSTS_DIR, filename), caption)
    except Exception as e:                           # never publish content that could not be checked
        os.remove(os.path.join(POSTS_DIR, filename))
        app.logger.exception('Analysis failed')
        flash(f'Could not analyse the image, so it was not posted ({e}).', 'error')
        return redirect(url_for('index'))

    post = Post(image=filename, caption=caption, user_id=user.id,
                analysis_json=json.dumps(analysis))
    db.session.add(post)

    if analysis['bullying_detected']:
        loss = round(analysis['combined_score'], 2)      # penalty = the bullying score itself (10 - 0.40 = 9.60)
        user.reputation_score = max(0.0, user.reputation_score - loss)
        flash(f'🚨 Cyberbullying detected in your post! Reputation decreased by {loss:.2f}. '
              f'Current score: {user.reputation_score:.1f}/10', 'warning')
        if user.is_blocked:
            flash('❌ ' + restriction_message(user), 'error')
    else:
        flash('✅ Post uploaded - no cyberbullying detected.', 'success')

    db.session.commit()
    session['last_post_id'] = post.id               # only the id goes in the cookie (it is size-limited)
    return redirect(url_for('index'))


# ============================================================
# Like / Share / Comment
# ============================================================
@app.route('/like/<int:post_id>', methods=['POST'])
def like_post(post_id):
    user = g.user
    get_or_404(Post, post_id)
    existing = Like.query.filter_by(user_id=user.id, post_id=post_id).first()
    if existing:
        db.session.delete(existing)
        liked = False
    else:
        db.session.add(Like(user_id=user.id, post_id=post_id))
        liked = True
    db.session.commit()
    return jsonify({'liked': liked, 'like_count': Like.query.filter_by(post_id=post_id).count()})


@app.route('/share/<int:post_id>', methods=['POST'])
def share_post(post_id):
    get_or_404(Post, post_id)
    db.session.add(Share(user_id=g.user.id, post_id=post_id))
    db.session.commit()
    return jsonify({'shared': True, 'share_count': Share.query.filter_by(post_id=post_id).count()})


@app.route('/comment/<int:post_id>', methods=['POST'])
def add_comment(post_id):
    user = g.user
    if user.is_blocked:
        return jsonify({'error': restriction_message(user), 'restricted': True}), 403
    if db.session.get(Post, post_id) is None:
        return jsonify({'error': 'Post not found'}), 404

    data = request.get_json(silent=True) or {}
    comment_text = str(data.get('comment', '')).strip()[:1000]
    if not comment_text:
        return jsonify({'error': 'Comment cannot be empty'}), 400

    try:
        ml.ensure_ready()
        # text toxicity + emoji analysis; emojis can only raise the score (never lower it)
        score, emoji = emoji_analysis.comment_score(comment_text, ml.toxicity_score(comment_text))
    except ml.ModelsUnavailable as e:
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        app.logger.exception('Comment analysis failed')
        return jsonify({'error': f'Could not analyse the comment: {e}'}), 500

    detected = score > COMMENT_BULLYING_THRESHOLD
    loss = 0.0
    if detected:
        loss = round(score, 2)                           # penalty = the bullying score itself
        user.reputation_score = max(0.0, user.reputation_score - loss)

    comment = Comment(text=comment_text, user_id=user.id, post_id=post_id, bullying_score=score)
    db.session.add(comment)
    db.session.commit()

    return jsonify({
        'success': True,
        'restricted': user.is_blocked,
        'comment': {
            'id': comment.id,
            'text': comment.text,
            'author': user.username,
            'created_at': comment.created_at.strftime('%b %d, %Y at %I:%M %p'),
            'bullying_detected': detected,
            'user_reputation': user.reputation_score,
            'reputation_loss': loss,
            'emoji': emoji_brief(emoji),
        },
    })


# ============================================================
# Post analysis page (reads the cached result)
# ============================================================
@app.route('/post/<int:post_id>/analysis')
def post_analysis(post_id):
    post = get_or_404(Post, post_id)
    try:
        a = get_post_analysis(post)
    except ml.ModelsUnavailable as e:
        flash(str(e), 'error')
        return redirect(url_for('index'))
    if a is None:
        flash('The image file for this post is missing.', 'error')
        return redirect(url_for('index'))
    return render_template(
        'post_analysis.html', post=post,
        clip_results=a.get('image_predictions'),
        clip_bullying_score=a['image_bullying_score'],
        clip_overall_label=a.get('clip_overall_label'),
        clip_overall_conf=a.get('clip_overall_conf', 0.0),
        combined_score=a['combined_score'],
        text_bullying_score=a['text_bullying_score'],
        ocr_text=a['ocr_detail'],
        ocr_confidence=a['ocr_confidence'],
        ocr_bullying_score=a['ocr_bullying_score'],
        image_predictions=a.get('image_predictions'),
        image_bullying_score=a['image_bullying_score'],
        emoji=a.get('emoji'),
    )


# ============================================================
# Profile / follow
# ============================================================
@app.route('/profile/<username>')
def profile(username):
    user = g.user
    profile_user = User.query.filter_by(username=username).first_or_404()
    is_own = profile_user.id == user.id
    is_following = has_pending = False
    can_view = is_own or not profile_user.is_private

    if not is_own:
        is_following = Follow.query.filter_by(follower_id=user.id, followed_id=profile_user.id).first() is not None
        if profile_user.is_private and not is_following:
            can_view = False
            has_pending = FollowRequest.query.filter_by(requester_id=user.id, target_id=profile_user.id,
                                                        status='pending').first() is not None

    posts = reels = []
    posts_count = Post.query.filter_by(user_id=profile_user.id).count()
    if can_view:
        posts = Post.query.filter_by(user_id=profile_user.id).order_by(Post.created_at.desc()).all()
        reels = Reel.query.filter_by(user_id=profile_user.id).order_by(Reel.created_at.desc()).all()

    return render_template(
        'profile.html', user=user, profile_user=profile_user, posts=posts, reels=reels,
        posts_count=posts_count,
        followers_count=Follow.query.filter_by(followed_id=profile_user.id).count(),
        following_count=Follow.query.filter_by(follower_id=profile_user.id).count(),
        is_following=is_following, is_own_profile=is_own, can_view_content=can_view,
        has_pending_request=has_pending)


@app.route('/follow/<int:user_id>', methods=['POST'])
def follow_user(user_id):
    user = g.user
    if user_id == user.id:
        return jsonify({'error': 'Cannot follow yourself'}), 400
    target = get_or_404(User, user_id)

    existing = Follow.query.filter_by(follower_id=user.id, followed_id=user_id).first()
    if existing:
        db.session.delete(existing)
        db.session.commit()
        return jsonify({'following': False})

    pending = FollowRequest.query.filter_by(requester_id=user.id, target_id=user_id, status='pending').first()
    if pending:                                      # cancelling a request is always allowed
        db.session.delete(pending)
        db.session.commit()
        return jsonify({'request_sent': False})

    if user.is_blocked:
        return jsonify({'error': 'Your account is restricted. You cannot follow other users.',
                        'restricted': True}), 403

    if target.is_private:
        db.session.add(FollowRequest(requester_id=user.id, target_id=user_id, status='pending'))
        db.session.commit()
        return jsonify({'request_sent': True})

    db.session.add(Follow(follower_id=user.id, followed_id=user_id))
    db.session.commit()
    return jsonify({'following': True})


@app.route('/handle-follow-request/<int:request_id>', methods=['POST'])
def handle_follow_request(request_id):
    user = g.user
    fr = get_or_404(FollowRequest, request_id)
    if fr.target_id != user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    action = (request.get_json(silent=True) or {}).get('action')
    if action == 'accept':
        if not Follow.query.filter_by(follower_id=fr.requester_id, followed_id=user.id).first():
            db.session.add(Follow(follower_id=fr.requester_id, followed_id=user.id))
        fr.status = 'accepted'
    elif action == 'reject':
        fr.status = 'rejected'
    else:
        return jsonify({'error': 'Invalid action'}), 400
    db.session.commit()
    return jsonify({'success': True})


# ============================================================
# Main pages
# ============================================================
def following_ids_of(user):
    ids = [r[0] for r in db.session.query(Follow.followed_id).filter_by(follower_id=user.id)]
    ids.append(user.id)
    return ids


@app.route('/')
def index():
    user = g.user
    ids = following_ids_of(user)
    posts = (Post.query.filter(Post.user_id.in_(ids))
             .options(selectinload(Post.author), selectinload(Post.likes),
                      selectinload(Post.comments).selectinload(Comment.author))
             .order_by(Post.created_at.desc()).limit(MAX_POSTS_PER_PAGE).all())
    suggestions = User.query.filter(User.id.notin_(ids)).limit(5).all()

    analysis = None
    last_id = session.pop('last_post_id', None)
    if last_id:
        post = db.session.get(Post, last_id)
        if post and post.user_id == user.id and post.analysis_json:
            analysis = json.loads(post.analysis_json)
            analysis['post_id'] = post.id
    return render_template('index.html', user=user, posts=posts, suggestions=suggestions,
                           analysis_results=analysis)


@app.route('/explore')
def explore():
    user = g.user
    ids = following_ids_of(user)
    base = Post.query.options(selectinload(Post.author))
    posts = base.filter(~Post.user_id.in_(ids)).order_by(db.func.random()).limit(20).all()
    if not posts:
        posts = base.order_by(db.func.random()).limit(20).all()
    return render_template('explore.html', user=user, posts=posts)


@app.route('/reels')
def reels():
    user = g.user
    ids = following_ids_of(user)
    items = (Reel.query.filter(Reel.user_id.in_(ids)).options(selectinload(Reel.author))
             .order_by(Reel.created_at.desc()).limit(30).all())
    return render_template('reels.html', user=user, reels=items)


@app.route('/liked-posts')
def liked_posts():
    user = g.user
    posts = (Post.query.join(Like).filter(Like.user_id == user.id)
             .options(selectinload(Post.author)).order_by(Like.created_at.desc()).all())
    return render_template('liked_posts.html', user=user, liked_posts=posts)


@app.route('/search')
def search():
    user = g.user
    q = request.args.get('q', '').strip()
    users, posts = [], []
    if q:
        like = f'%{q}%'
        users = User.query.filter(User.username.ilike(like) | User.full_name.ilike(like)).limit(50).all()
        posts = (Post.query.filter(Post.caption.ilike(like)).options(selectinload(Post.author))
                 .order_by(Post.created_at.desc()).limit(50).all())
    return render_template('search.html', user=user, users=users, posts=posts, query=q)


@app.route('/follow-requests')
def follow_requests():
    user = g.user
    if not user.is_private:
        return redirect(url_for('index'))
    reqs = (FollowRequest.query.filter_by(target_id=user.id, status='pending')
            .options(selectinload(FollowRequest.requester)).all())
    return render_template('follow_requests.html', user=user, requests=reqs)


@app.route('/settings', methods=['GET', 'POST'])
def settings():
    user = g.user
    if request.method == 'POST':
        email = request.form.get('email', user.email).strip().lower()
        if email != user.email and User.query.filter(User.email == email, User.id != user.id).first():
            flash('That email is already in use', 'error')
            return redirect(url_for('settings'))

        user.full_name = request.form.get('full_name', user.full_name).strip() or user.full_name
        user.bio = request.form.get('bio', user.bio or '')[:300]
        user.email = email
        user.is_private = request.form.get('is_private') == 'on'

        pic = request.files.get('profile_pic')
        if pic and pic.filename:
            if not allowed_file(pic.filename, {'png', 'jpg', 'jpeg', 'gif', 'webp'}):
                flash('Profile picture must be an image', 'error')
            else:
                try:
                    user.profile_pic = save_image(pic, PROFILES_DIR, max_side=400,
                                                  prefix=f'profile_{user.id}_')
                except ValueError as e:
                    flash(str(e), 'error')

        db.session.commit()
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('settings'))
    return render_template('settings.html', user=user)


# ============================================================
# Errors
# ============================================================
@app.errorhandler(404)
def not_found_error(_e):
    return render_template('404.html'), 404


@app.errorhandler(413)
def too_large(_e):
    flash('That file is too large (max 16 MB).', 'error')
    return redirect(url_for('index'))


@app.errorhandler(500)
def internal_error(_e):
    db.session.rollback()
    return render_template('500.html'), 500


# ============================================================
# Run
# ============================================================
with app.app_context():
    ensure_folders_and_files()
    ensure_schema()

if __name__ == '__main__':
    ml.hub.start()      # models load in the background; the site is usable immediately
    app.run(host='127.0.0.1', port=5000,
            debug=os.environ.get('FLASK_DEBUG') == '1', use_reloader=False, threaded=True)
