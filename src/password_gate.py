from __future__ import annotations

import hashlib
import os


def _password_hash() -> str:
    explicit_hash = os.environ.get("SITE_PASSWORD_HASH", "").strip()
    if explicit_hash:
        return explicit_hash.lower()

    password = os.environ.get("SITE_PASSWORD", "").strip()
    if not password:
        return ""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def gate_enabled() -> bool:
    return bool(_password_hash())


def gate_head() -> str:
    password_hash = _password_hash()
    if not password_hash:
        return ""

    return f"""
<style>
  #password-gate {{
    position: fixed;
    inset: 0;
    z-index: 9999;
    display: grid;
    place-items: center;
    background: #f7f7f4;
    color: #1c1c1c;
    padding: 1rem;
  }}
  #password-gate form {{
    width: min(100%, 22rem);
    border: 1px solid #d8d8d2;
    background: #fff;
    padding: 1rem;
  }}
  #password-gate label {{
    display: block;
    font-family: Helvetica, Arial, sans-serif;
    font-size: 0.8rem;
    margin-bottom: 0.35rem;
    color: #555;
  }}
  #password-gate input {{
    box-sizing: border-box;
    width: 100%;
    padding: 0.55rem;
    border: 1px solid #bbb;
    font: inherit;
  }}
  #password-gate button {{
    margin-top: 0.7rem;
    padding: 0.5rem 0.75rem;
    border: 1px solid #333;
    background: #222;
    color: #fff;
    font: inherit;
    cursor: pointer;
  }}
  #password-error {{
    min-height: 1.2rem;
    margin: 0.6rem 0 0;
    color: #9b1c1c;
    font-size: 0.9rem;
  }}
</style>
<script>
  window.READING_NOTES_PASSWORD_HASH = "{password_hash}";
</script>
"""


def protect_html_document(text: str) -> str:
    if not gate_enabled() or "READING_NOTES_PASSWORD_HASH" in text:
        return text

    if "</head>" in text:
        text = text.replace("</head>", gate_head() + "\n</head>", 1)
    elif "<html" in text:
        html_start = text.find("<html")
        html_end = text.find(">", html_start)
        if html_end != -1:
            text = text[: html_end + 1] + "\n<head>" + gate_head() + "</head>" + text[html_end + 1 :]
        else:
            text = "<head>" + gate_head() + "</head>\n" + text
    else:
        text = "<head>" + gate_head() + "</head>\n" + text

    if "<body>" in text:
        text = text.replace("<body>", "<body>\n" + gate_body_start(), 1)
    elif "<body " in text:
        body_start = text.find("<body ")
        body_end = text.find(">", body_start)
        if body_end != -1:
            text = text[: body_end + 1] + "\n" + gate_body_start() + text[body_end + 1 :]
    else:
        text = gate_body_start() + text

    if "</body>" in text:
        text = text.replace("</body>", gate_body_end() + "\n</body>", 1)
    else:
        text = text + gate_body_end()

    return text


def gate_body_start() -> str:
    if not _password_hash():
        return ""

    return """
<div id="password-gate">
  <form id="password-form">
    <label for="site-password">Password</label>
    <input id="site-password" type="password" autocomplete="current-password" autofocus>
    <button type="submit">Enter</button>
    <p id="password-error" aria-live="polite"></p>
  </form>
</div>
<main id="protected-content" hidden>
"""


def gate_body_end() -> str:
    if not _password_hash():
        return ""

    return """
</main>
<script>
  (function () {
    const expectedHash = window.READING_NOTES_PASSWORD_HASH || "";
    const gate = document.getElementById("password-gate");
    const content = document.getElementById("protected-content");
    const form = document.getElementById("password-form");
    const input = document.getElementById("site-password");
    const error = document.getElementById("password-error");

    async function sha256(value) {
      const bytes = new TextEncoder().encode(value);
      const digest = await crypto.subtle.digest("SHA-256", bytes);
      return Array.from(new Uint8Array(digest))
        .map((byte) => byte.toString(16).padStart(2, "0"))
        .join("");
    }

    function unlock() {
      gate.remove();
      content.hidden = false;
    }

    if (localStorage.getItem("reading-notes-unlocked") === expectedHash) {
      unlock();
      return;
    }

    form.addEventListener("submit", async function (event) {
      event.preventDefault();
      const enteredHash = await sha256(input.value);
      if (enteredHash === expectedHash) {
        localStorage.setItem("reading-notes-unlocked", expectedHash);
        unlock();
      } else {
        error.textContent = "Wrong password.";
        input.select();
      }
    });
  }());
</script>
"""
