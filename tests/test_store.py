import json
from mycoder.memory.store import SessionStore


def test_new_session(tmp_path):
    store = SessionStore(base_dir=tmp_path)
    session = store.new_session(model="test-model", cwd="/test")
    assert session["model"] == "test-model"
    assert session["cwd"] == "/test"
    assert session["messages"] == []
    assert "id" in session
    assert "created" in session


def test_save_and_load(tmp_path):
    store = SessionStore(base_dir=tmp_path)
    session = store.new_session(model="test-model", cwd="/test")
    session["messages"].append({"role": "user", "content": "hello"})
    store.save(session)

    loaded = store.load(session["id"])
    assert loaded["messages"] == [{"role": "user", "content": "hello"}]


def test_load_latest(tmp_path):
    store = SessionStore(base_dir=tmp_path)

    s1 = store.new_session(model="m", cwd="/a")
    store.save(s1)

    s2 = store.new_session(model="m", cwd="/b")
    s2["messages"].append({"role": "user", "content": "latest"})
    store.save(s2)

    latest = store.load_latest()
    assert latest is not None
    assert latest["messages"] == [{"role": "user", "content": "latest"}]


def test_load_latest_empty(tmp_path):
    store = SessionStore(base_dir=tmp_path)
    assert store.load_latest() is None
