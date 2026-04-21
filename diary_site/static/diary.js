(function () {
  function buildMarkdownEngine() {
    if (!window.markdownit) return null;
    const md = window.markdownit({
      html: false,
      linkify: true,
      breaks: true,
      typographer: true,
    });

    if (window.markdownitKatex) {
      md.use(window.markdownitKatex);
    }
    return md;
  }

  function sanitizeHtml(html) {
    if (!window.DOMPurify) return html;
    return window.DOMPurify.sanitize(html, {
      USE_PROFILES: { html: true, svg: true, mathMl: true },
    });
  }

  async function copyText(text) {
    if (!navigator.clipboard || !navigator.clipboard.writeText) {
      throw new Error("Clipboard API not available");
    }
    await navigator.clipboard.writeText(text);
  }

  function attachCopyButtons(root) {
    const blocks = root.querySelectorAll("pre > code");
    blocks.forEach((code) => {
      const pre = code.parentElement;
      if (!pre || pre.querySelector(".copy-code-btn")) return;

      pre.classList.add("code-block");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "copy-code-btn";
      btn.textContent = "复制";

      btn.addEventListener("click", async () => {
        const text = code.textContent || "";
        if (!text) return;
        try {
          await copyText(text);
          btn.textContent = "已复制";
          window.setTimeout(() => {
            btn.textContent = "复制";
          }, 1200);
        } catch (_err) {
          btn.textContent = "失败";
          window.setTimeout(() => {
            btn.textContent = "复制";
          }, 1200);
        }
      });

      pre.appendChild(btn);
    });
  }

  function renderMarkdownEntries() {
    const md = buildMarkdownEngine();
    const cards = document.querySelectorAll(".diary-entry");

    cards.forEach((card) => {
      const source = card.querySelector(".md-source");
      const target = card.querySelector(".js-markdown-render");
      if (!source || !target) return;

      const raw = source.value || "";
      if (!raw.trim()) {
        target.innerHTML = '<p class="meta">(空内容)</p>';
        return;
      }

      if (!md) {
        const escaped = raw
          .replaceAll("&", "&amp;")
          .replaceAll("<", "&lt;")
          .replaceAll(">", "&gt;")
          .replaceAll('"', "&quot;");
        target.innerHTML = `<pre>${escaped}</pre>`;
        return;
      }

      const rendered = md.render(raw);
      target.innerHTML = sanitizeHtml(rendered);
      attachCopyButtons(target);
    });
  }

  renderMarkdownEntries();
})();
