"""Optional: the app now creates these folders (and a real default avatar) automatically on start-up."""
import os

BASE = os.path.dirname(os.path.abspath(__file__))
for d in ('static/uploads/posts', 'static/uploads/profiles', 'static/uploads/reels', 'templates'):
    os.makedirs(os.path.join(BASE, d), exist_ok=True)
    print('Ready:', d)
