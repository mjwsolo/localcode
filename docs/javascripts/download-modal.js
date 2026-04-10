document.addEventListener("DOMContentLoaded", () => {
  const headerInner = document.querySelector(".md-header__inner");
  if (!headerInner || document.querySelector(".download-trigger")) return;
  const parts = window.location.pathname.split("/").filter(Boolean);
  const basePath = parts.length > 0 && parts[0] === "localcode" ? "/localcode/" : "/";

  const siteNav = document.createElement("nav");
  siteNav.className = "site-nav";
  siteNav.innerHTML = `
    <a class="site-nav__brand" href="${basePath}">localcode</a>
    <a class="site-nav__link" href="${basePath}enterprise/">Enterprise</a>
    <span class="site-nav__link site-nav__link--muted">Docs <em>coming soon</em></span>
  `;

  const trigger = document.createElement("button");
  trigger.className = "download-trigger";
  trigger.type = "button";
  trigger.textContent = "Download";

  const overlay = document.createElement("div");
  overlay.className = "download-modal";
  overlay.hidden = true;
  overlay.innerHTML = `
    <div class="download-modal__backdrop" data-close="true"></div>
    <div class="download-modal__panel" role="dialog" aria-modal="true" aria-labelledby="download-modal-title">
      <div class="download-modal__header">
        <h3 id="download-modal-title">Install LocalCode</h3>
        <button type="button" class="download-modal__close" aria-label="Close">×</button>
      </div>
      <p class="download-modal__copy">Run this in your terminal:</p>
      <div class="download-modal__command">
        <code>pip install localcode && localcode</code>
        <button type="button" class="download-modal__copybtn">Copy</button>
      </div>
    </div>
  `;

  const openModal = () => {
    overlay.hidden = false;
    document.body.classList.add("download-modal-open");
  };

  const closeModal = () => {
    overlay.hidden = true;
    document.body.classList.remove("download-modal-open");
  };

  trigger.addEventListener("click", openModal);
  overlay.addEventListener("click", (event) => {
    if (event.target instanceof HTMLElement && event.target.dataset.close === "true") {
      closeModal();
    }
  });
  overlay.querySelector(".download-modal__close")?.addEventListener("click", closeModal);

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !overlay.hidden) closeModal();
  });

  overlay.querySelector(".download-modal__copybtn")?.addEventListener("click", async (event) => {
    const button = event.currentTarget;
    if (!(button instanceof HTMLButtonElement)) return;
    const command = "pip install localcode && localcode";
    try {
      await navigator.clipboard.writeText(command);
      button.textContent = "Copied";
      window.setTimeout(() => {
        button.textContent = "Copy";
      }, 1200);
    } catch {
      button.textContent = "Copy failed";
      window.setTimeout(() => {
        button.textContent = "Copy";
      }, 1200);
    }
  });

  headerInner.appendChild(siteNav);
  headerInner.appendChild(trigger);
  document.body.appendChild(overlay);

  const newsletterForm = document.querySelector("[data-newsletter-form]");
  newsletterForm?.addEventListener("submit", (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    if (!(form instanceof HTMLFormElement)) return;
    const emailInput = form.querySelector('input[name="email"]');
    if (!(emailInput instanceof HTMLInputElement)) return;
    const email = emailInput.value.trim();
    if (!email) return;
    const subject = encodeURIComponent("LocalCode newsletter signup");
    const body = encodeURIComponent(`Please add this email to the LocalCode newsletter / waitlist:\n\n${email}`);
    window.location.href = `mailto:the maintainers (see SECURITY.md)?subject=${subject}&body=${body}`;
  });

  const demoTabs = Array.from(document.querySelectorAll("[data-demo-tab]"));
  const demoPanels = Array.from(document.querySelectorAll("[data-demo-panel]"));
  for (const tab of demoTabs) {
    tab.addEventListener("click", () => {
      const key = tab.getAttribute("data-demo-tab");
      for (const other of demoTabs) {
        other.classList.toggle("is-active", other === tab);
      }
      for (const panel of demoPanels) {
        panel.classList.toggle("is-active", panel.getAttribute("data-demo-panel") === key);
      }
    });
  }
});
