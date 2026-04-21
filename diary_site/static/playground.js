
(function () {
  const boot = window.IDE_BOOTSTRAP || {};
  const limits = boot.limits || { maxCodeChars: 20000, maxStdinChars: 8000, maxFileChars: 250000 };
  const $ = (id) => document.getElementById(id);

  const runtimeEnvSelect = $("runtime-env-select");
  const refreshEnvBtn = $("refresh-env-btn");
  const runBtn = $("run-btn");
  const saveBtn = $("save-btn");
  const newFileBtn = $("new-file-btn");
  const newFolderBtn = $("new-folder-btn");
  const renameItemBtn = $("rename-item-btn");
  const deleteItemBtn = $("delete-item-btn");

  const terminalStartBtn = $("terminal-start-btn");
  const shellStartBtn = $("shell-start-btn");
  const terminalStopBtn = $("terminal-stop-btn");
  const terminalSendBtn = $("terminal-send-btn");
  const terminalInputEl = $("terminal-input");
  const terminalOutputEl = $("terminal-output");
  const terminalStateEl = $("terminal-state");

  const cwdInput = $("cwd-input");
  const openDirBtn = $("open-dir-btn");
  const upDirBtn = $("up-dir-btn");
  const homeDirBtn = $("home-dir-btn");
  const fileListEl = $("file-list");
  const fileCountEl = $("file-count");

  const editorTabsEl = $("editor-tabs");
  const currentFileNameEl = $("current-file-name");
  const saveStateEl = $("save-state");
  const codeEditorHost = $("code-editor");
  const codeEditorFallback = $("code-editor-fallback");
  const stdinBox = $("stdin-box");
  const expectedBox = $("expected-box");

  const verdictBadge = $("verdict-badge");
  const stdoutView = $("stdout-view");
  const stderrView = $("stderr-view");
  const langTabs = Array.from(document.querySelectorAll(".lang-tab"));

  const searchPanelEl = $("search-panel");
  const searchInputEl = $("search-input");
  const searchBtnEl = $("search-btn");
  const searchResultsEl = $("search-results");

  const gitPanelEl = $("git-panel");
  const gitRefreshBtn = $("git-refresh-btn");
  const gitBranchEl = $("git-branch");
  const gitStatusListEl = $("git-status-list");
  const gitCommitMessageEl = $("git-commit-message");
  const gitCommitBtn = $("git-commit-btn");
  const gitDiffViewEl = $("git-diff-view");

  const commandOverlay = $("command-palette");
  const commandInputEl = $("command-input");
  const commandListEl = $("command-list");
  const quickOpenOverlay = $("quick-open");
  const quickOpenInputEl = $("quick-open-input");
  const quickOpenListEl = $("quick-open-list");

  const toggleCommandBtn = $("toggle-command-btn");
  const toggleQuickOpenBtn = $("toggle-quickopen-btn");
  const toggleSearchBtn = $("toggle-search-btn");
  const toggleGitBtn = $("toggle-git-btn");

  const cpuLoadEl = $("cpu-load");
  const memUsedEl = $("mem-used");
  const memTotalEl = $("mem-total");
  const diskUsedEl = $("disk-used");
  const diskTotalEl = $("disk-total");
  const diskPathEl = $("disk-path");
  const statusTimeEl = $("status-time");
  const statusErrorEl = $("status-error");
  const refreshStatusBtn = $("refresh-status-btn");

  let entries = [];
  let currentDir = String(boot.initialDir || "");
  let rootDir = String(boot.rootDir || "");
  let homeDir = String(boot.homeDir || "");
  let parentDir = "";
  let selectedPath = "";
  let currentLanguage = "python";

  let monacoEditor = null;
  let usingMonaco = false;
  let syncingModel = false;

  const tabs = [];
  const cacheByPath = new Map();
  const modelByPath = new Map();
  let activeTabPath = "";

  let terminalSessionId = "";
  let terminalCursor = 0;
  let terminalBuffer = "";
  let terminalPollTimer = null;
  let statusTimer = null;

  function npath(p) { return String(p || "").replace(/\\/g, "/").replace(/\/+$/, ""); }
  function base(p) { const x = npath(p); const i = x.lastIndexOf("/"); return i >= 0 ? x.slice(i + 1) : x; }
  function dir(p) { const x = npath(p); const i = x.lastIndexOf("/"); return i >= 0 ? x.slice(0, i) : ""; }
  function isChild(pathText, parentText) { const p = npath(pathText); const q = npath(parentText); return !!(p && q && (p === q || p.startsWith(`${q}/`))); }

  function showError(msg) { stderrView.style.display = "block"; stderrView.textContent = String(msg || "操作失败"); }
  function clearError() { stderrView.style.display = "none"; stderrView.textContent = ""; }

  function setVerdict(text) {
    verdictBadge.textContent = text || "Waiting";
    verdictBadge.className = "verdict";
    const t = String(text || "").toLowerCase();
    if (t.includes("accepted") || t.includes("saved")) verdictBadge.classList.add("accepted");
    else if (t.includes("wrong") || t.includes("time") || t.includes("stopped")) verdictBadge.classList.add("warning");
    else if (t.includes("error") || t.includes("fail") || t.includes("compilation")) verdictBadge.classList.add("error");
    else verdictBadge.classList.add("waiting");
  }

  function setTerminalState(text, type) {
    terminalStateEl.textContent = text;
    terminalStateEl.className = "verdict";
    terminalStateEl.classList.add(type || "waiting");
  }

  function getTab(pathText) { const p = npath(pathText); return tabs.find((t) => npath(t.path) === p) || null; }

  function setLanguage(lang) {
    currentLanguage = lang === "cpp" ? "cpp" : "python";
    langTabs.forEach((tab) => tab.classList.toggle("is-active", tab.dataset.lang === currentLanguage));
    if (usingMonaco && monacoEditor && monacoEditor.getModel()) window.monaco.editor.setModelLanguage(monacoEditor.getModel(), currentLanguage);
  }

  function detectLanguage(pathText) {
    const p = String(pathText || "").toLowerCase();
    return p.endsWith(".cpp") || p.endsWith(".cc") || p.endsWith(".cxx") || p.endsWith(".hpp") || p.endsWith(".h") ? "cpp" : "python";
  }

  function renderTabs() {
    editorTabsEl.replaceChildren();
    if (!tabs.length) {
      const empty = document.createElement("div");
      empty.className = "editor-tab-empty";
      empty.textContent = "未打开文件";
      editorTabsEl.appendChild(empty);
      currentFileNameEl.textContent = "未选择文件";
      saveStateEl.textContent = "已保存";
      return;
    }
    for (const tab of tabs) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `editor-tab${npath(tab.path) === npath(activeTabPath) ? " is-active" : ""}`;
      btn.dataset.path = tab.path;
      btn.innerHTML = `<span class="editor-tab-name">${tab.name}${tab.dirty ? " *" : ""}</span><span class="editor-tab-close" data-action="close" data-path="${tab.path}">×</span>`;
      editorTabsEl.appendChild(btn);
    }
    const active = getTab(activeTabPath);
    if (active) {
      currentFileNameEl.textContent = active.path;
      saveStateEl.textContent = active.dirty ? "未保存" : "已保存";
    }
  }

  function markTabDirty(pathText, dirty) {
    const tab = getTab(pathText);
    if (!tab) return;
    tab.dirty = !!dirty;
    renderTabs();
  }

  function setupFallbackEditor() {
    usingMonaco = false;
    codeEditorHost.style.display = "none";
    codeEditorFallback.style.display = "block";
    codeEditorFallback.addEventListener("input", () => {
      if (!activeTabPath) return;
      cacheByPath.set(activeTabPath, codeEditorFallback.value);
      markTabDirty(activeTabPath, true);
    });
  }

  function monacoUri(pathText) {
    return window.monaco.Uri.parse(`file:///${encodeURIComponent(npath(pathText)).replace(/%2F/g, "/")}`);
  }

  function ensureModel(pathText, content, language) {
    if (!usingMonaco) return null;
    const p = npath(pathText);
    let model = modelByPath.get(p);
    if (!model) {
      model = window.monaco.editor.createModel(content, language || detectLanguage(p), monacoUri(p));
      modelByPath.set(p, model);
      model.onDidChangeContent(() => {
        if (syncingModel) return;
        cacheByPath.set(p, model.getValue());
        markTabDirty(p, true);
      });
    }
    return model;
  }

  function switchTab(pathText, checkDirty) {
    const next = npath(pathText);
    const cur = getTab(activeTabPath);
    if (checkDirty && cur && cur.dirty && npath(cur.path) !== next) {
      if (!window.confirm(`当前文件 ${cur.name} 有未保存改动，确定继续切换吗？`)) return;
    }
    const tab = getTab(next);
    if (!tab) return;

    activeTabPath = tab.path;
    selectedPath = tab.path;
    setLanguage(tab.language || detectLanguage(tab.path));

    const text = cacheByPath.get(tab.path) || "";
    if (usingMonaco) {
      const model = ensureModel(tab.path, text, tab.language || detectLanguage(tab.path));
      if (model) monacoEditor.setModel(model);
    } else {
      codeEditorFallback.value = text;
    }

    renderTabs();
    renderEntryList();
  }

  function closeTab(pathText) {
    const tab = getTab(pathText);
    if (!tab) return;
    if (tab.dirty && !window.confirm(`文件 ${tab.name} 有未保存改动，确认关闭？`)) return;

    const p = npath(tab.path);
    const idx = tabs.findIndex((item) => npath(item.path) === p);
    if (idx < 0) return;
    tabs.splice(idx, 1);

    cacheByPath.delete(p);
    const model = modelByPath.get(p);
    if (model) {
      modelByPath.delete(p);
      model.dispose();
    }

    if (npath(activeTabPath) === p) {
      const next = tabs[idx] || tabs[idx - 1] || null;
      activeTabPath = next ? next.path : "";
      if (next) switchTab(next.path, false);
      else {
        if (usingMonaco && monacoEditor) monacoEditor.setModel(null);
        else codeEditorFallback.value = "";
      }
    }
    renderTabs();
  }

  function getEditorValue() {
    if (!activeTabPath) return "";
    if (usingMonaco && monacoEditor && monacoEditor.getModel()) return monacoEditor.getModel().getValue();
    return cacheByPath.get(activeTabPath) || codeEditorFallback.value || "";
  }

  function upsertTab(pathText, content, language) {
    const p = npath(pathText);
    let tab = getTab(p);
    if (!tab) {
      tab = { path: p, name: base(p), language: language || detectLanguage(p), dirty: false };
      tabs.push(tab);
    }

    const text = String(content || "");
    cacheByPath.set(p, text);
    if (usingMonaco) {
      const model = ensureModel(p, text, tab.language);
      if (model && model.getValue() !== text) {
        syncingModel = true;
        model.setValue(text);
        syncingModel = false;
      }
    }
    tab.dirty = false;
    switchTab(p, false);
  }

  async function openFile(pathText) {
    const known = getTab(pathText);
    if (known) {
      switchTab(known.path, true);
      return;
    }
    const response = await fetch(`/api/ide/file?path=${encodeURIComponent(pathText)}&cwd=${encodeURIComponent(currentDir)}`);
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.error || "打开文件失败");
    upsertTab(data.path || pathText, data.content || "", data.language || detectLanguage(data.path || pathText));
  }

  async function saveCurrentTab() {
    const tab = getTab(activeTabPath);
    if (!tab) throw new Error("请先打开文件");
    const content = getEditorValue();
    if (content.length > limits.maxFileChars) throw new Error(`文件内容超过 ${limits.maxFileChars} 字符`);

    saveBtn.disabled = true;
    try {
      const response = await fetch("/api/ide/file", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: tab.path, cwd: currentDir, content }),
      });
      const data = await response.json();
      if (!response.ok || !data.ok) throw new Error(data.error || "保存失败");
      cacheByPath.set(tab.path, content);
      markTabDirty(tab.path, false);
      await loadDirectory(currentDir);
      setVerdict("Saved");
    } finally {
      saveBtn.disabled = false;
    }
  }

  async function createItem(type) {
    const defaultName = type === "folder" ? "new-folder" : currentLanguage === "cpp" ? "main.cpp" : "main.py";
    const input = window.prompt(type === "folder" ? "请输入新目录名或路径" : "请输入新文件名或路径", defaultName);
    if (!input) return;

    const response = await fetch("/api/ide/new-item", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type, path: input.trim(), cwd: currentDir }),
    });
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.error || "创建失败");

    await loadDirectory(currentDir);
    if (type === "file") await openFile(data.path || input.trim());
  }

  async function renamePath(pathText) {
    const oldPath = npath(pathText || selectedPath);
    if (!oldPath) throw new Error("请先选择文件或目录");

    const nextName = window.prompt("输入新的文件名/目录名", base(oldPath));
    if (!nextName || !nextName.trim()) return;
    const newPath = nextName.includes("/") || nextName.includes("\\") ? nextName.trim() : `${dir(oldPath)}/${nextName.trim()}`;

    const response = await fetch("/api/ide/rename-item", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ old_path: oldPath, new_path: newPath, cwd: currentDir }),
    });
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.error || "重命名失败");

    const nextRoot = npath(data.path || newPath);
    for (const tab of tabs) {
      const p = npath(tab.path);
      if (p === oldPath || p.startsWith(`${oldPath}/`)) {
        const suffix = p.slice(oldPath.length);
        const newTabPath = `${nextRoot}${suffix}`;
        const cached = cacheByPath.get(tab.path);
        const oldModelPath = npath(tab.path);
        const oldModel = modelByPath.get(oldModelPath);
        cacheByPath.delete(tab.path);
        tab.path = newTabPath;
        tab.name = base(newTabPath);
        if (cached !== undefined) cacheByPath.set(newTabPath, cached);
        if (oldModel) {
          modelByPath.delete(oldModelPath);
          oldModel.dispose();
          ensureModel(newTabPath, cached || "", tab.language || detectLanguage(newTabPath));
        }
      }
    }
    if (activeTabPath && (npath(activeTabPath) === oldPath || npath(activeTabPath).startsWith(`${oldPath}/`))) {
      activeTabPath = `${nextRoot}${npath(activeTabPath).slice(oldPath.length)}`;
    }

    selectedPath = nextRoot;
    renderTabs();
    await loadDirectory(currentDir);
  }

  async function deletePath(pathText) {
    const target = npath(pathText || selectedPath);
    if (!target) throw new Error("请先选择文件或目录");
    if (!window.confirm(`确认删除：${target} ?`)) return;

    const response = await fetch("/api/ide/delete-item", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: target, cwd: currentDir }),
    });
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.error || "删除失败");

    for (const tab of [...tabs]) {
      const p = npath(tab.path);
      if (p === target || p.startsWith(`${target}/`)) closeTab(tab.path);
    }
    selectedPath = "";
    await loadDirectory(currentDir);
  }

  function renderEntryList() {
    const folderCount = entries.filter((i) => i.type === "dir").length;
    const fileCount = entries.filter((i) => i.type === "file").length;
    fileCountEl.textContent = `${fileCount} 文件 · ${folderCount} 文件夹`;

    fileListEl.replaceChildren();
    if (parentDir) {
      const parent = document.createElement("li");
      parent.className = "tree-row";
      parent.innerHTML = `<button type="button" class="tree-item folder-item" data-action="open" data-kind="dir" data-path="${parentDir}"><span class="tree-caret">↩</span><span class="tree-label">.. (上级目录)</span></button>`;
      fileListEl.appendChild(parent);
    }

    if (!entries.length) {
      const empty = document.createElement("li");
      empty.className = "file-empty";
      empty.textContent = "此目录为空";
      fileListEl.appendChild(empty);
      return;
    }

    for (const entry of entries) {
      const active = npath(selectedPath) === npath(entry.path) || npath(activeTabPath) === npath(entry.path);
      const li = document.createElement("li");
      li.className = "tree-row";
      li.innerHTML = `
        <button type="button" class="tree-item ${entry.type === "dir" ? "folder-item" : "file-item"}${active ? " is-active" : ""}" data-action="open" data-kind="${entry.type}" data-path="${entry.path}">
          <span class="tree-caret">${entry.type === "dir" ? "D" : "F"}</span>
          <span class="tree-label">${entry.type === "dir" ? `${entry.name}/` : entry.name}</span>
          ${entry.type === "file" ? `<span class="tree-meta">${typeof entry.size === "number" ? `${entry.size} B` : ""}</span>` : ""}
        </button>
        <span class="tree-actions">
          <button type="button" class="tree-mini-btn" data-action="rename" data-path="${entry.path}">重命名</button>
          <button type="button" class="tree-mini-btn danger" data-action="delete" data-path="${entry.path}">删除</button>
        </span>
      `;
      fileListEl.appendChild(li);
    }
  }

  async function loadDirectory(targetDir) {
    const query = targetDir ? `?dir=${encodeURIComponent(targetDir)}` : "";
    const response = await fetch(`/api/ide/files${query}`);
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.error || "打开目录失败");

    currentDir = String(data.cwd || "");
    rootDir = String(data.root_dir || rootDir || "");
    homeDir = String(data.home_dir || homeDir || "");
    parentDir = String(data.parent_dir || "");
    entries = Array.isArray(data.entries) ? data.entries : [];

    cwdInput.value = currentDir;
    upDirBtn.disabled = !parentDir;
    renderEntryList();
  }

  function applyRuntimeEnvs(envs, currentEnv) {
    runtimeEnvSelect.innerHTML = (envs || []).map((env) => `<option value="${env.id}">${env.label}</option>`).join("");
    runtimeEnvSelect.value = currentEnv || (envs && envs[0] ? envs[0].id : "system");
  }

  async function refreshRuntimeEnvs() {
    refreshEnvBtn.disabled = true;
    try {
      const response = await fetch("/api/runtime-envs?refresh=1");
      const data = await response.json();
      if (!response.ok || !data.ok) throw new Error(data.error || "获取环境失败");
      applyRuntimeEnvs(data.envs || [], data.current_env || "system");
    } finally {
      refreshEnvBtn.disabled = false;
    }
  }

  async function searchWorkspace(keyword) {
    const q = String(keyword || "").trim();
    if (q.length < 2) return [];
    const response = await fetch(`/api/ide/search?q=${encodeURIComponent(q)}&dir=${encodeURIComponent(currentDir)}`);
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.error || "搜索失败");
    return Array.isArray(data.results) ? data.results : [];
  }

  function renderSearchList(container, results, onOpen) {
    container.replaceChildren();
    if (!results.length) {
      const empty = document.createElement("li");
      empty.className = "search-empty";
      empty.textContent = "无结果";
      container.appendChild(empty);
      return;
    }

    for (const item of results) {
      const li = document.createElement("li");
      li.className = "search-result-item";
      li.innerHTML = `<button type="button" class="search-result-btn" data-path="${item.path}"><span>${item.name || base(item.path)}</span><small>${item.path}</small>${item.preview ? `<em>${item.preview}</em>` : ""}</button>`;
      container.appendChild(li);
    }

    container.querySelectorAll("button[data-path]").forEach((btn) => {
      btn.addEventListener("click", async () => onOpen(btn.dataset.path || ""));
    });
  }

  async function runGlobalSearch() {
    const results = await searchWorkspace(searchInputEl.value.trim());
    renderSearchList(searchResultsEl, results, async (path) => {
      searchPanelEl.classList.add("is-hidden");
      await openFile(path);
    });
  }

  async function runQuickOpenSearch() {
    const results = await searchWorkspace(quickOpenInputEl.value.trim());
    renderSearchList(quickOpenListEl, results, async (path) => {
      quickOpenOverlay.classList.add("is-hidden");
      await openFile(path);
    });
  }

  async function refreshGitStatus() {
    const response = await fetch(`/api/ide/git-status?cwd=${encodeURIComponent(currentDir)}`);
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.error || "Git 状态读取失败");

    gitStatusListEl.replaceChildren();
    gitDiffViewEl.textContent = "选择变更文件可查看 diff";
    if (!data.repo_found) {
      gitBranchEl.textContent = "当前目录不是 Git 仓库";
      const li = document.createElement("li");
      li.className = "search-empty";
      li.textContent = data.error || "未检测到 Git 仓库";
      gitStatusListEl.appendChild(li);
      return;
    }

    gitBranchEl.textContent = `Branch: ${data.branch || "(detached)"} · ${data.changed.length} changed`;
    if (!data.changed.length) {
      const li = document.createElement("li");
      li.className = "search-empty";
      li.textContent = "工作区干净，没有变更";
      gitStatusListEl.appendChild(li);
      return;
    }

    for (const item of data.changed) {
      const li = document.createElement("li");
      li.className = "search-result-item";
      li.innerHTML = `<button type="button" class="search-result-btn" data-git-path="${item.path}"><span>[${item.xy}] ${item.path}</span></button>`;
      gitStatusListEl.appendChild(li);
    }

    gitStatusListEl.querySelectorAll("button[data-git-path]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const target = btn.dataset.gitPath || "";
        const diffRes = await fetch(`/api/ide/git-diff?cwd=${encodeURIComponent(currentDir)}&path=${encodeURIComponent(target)}`);
        const diffData = await diffRes.json();
        if (!diffRes.ok || !diffData.ok) {
          gitDiffViewEl.textContent = diffData.error || "读取 diff 失败";
          return;
        }
        gitDiffViewEl.textContent = diffData.diff || "(无差异输出)";
      });
    });
  }

  async function commitGitChanges() {
    const message = gitCommitMessageEl.value.trim();
    if (!message) throw new Error("请输入 commit message");

    const response = await fetch("/api/ide/git-commit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cwd: currentDir, message, add_all: true }),
    });
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.error || "提交失败");

    gitCommitMessageEl.value = "";
    gitDiffViewEl.textContent = data.output || "Commit finished";
    await refreshGitStatus();
  }

  function appendTerminalOutput(chunk) {
    if (!chunk) return;
    terminalBuffer += String(chunk);
    if (terminalBuffer.length > 120000) terminalBuffer = terminalBuffer.slice(-120000);
    terminalOutputEl.textContent = terminalBuffer;
    terminalOutputEl.scrollTop = terminalOutputEl.scrollHeight;
  }

  function clearTerminalOutput() {
    terminalBuffer = "";
    terminalCursor = 0;
    terminalOutputEl.textContent = "";
  }

  function clearTerminalPoll() {
    if (terminalPollTimer) {
      window.clearInterval(terminalPollTimer);
      terminalPollTimer = null;
    }
  }

  async function pollTerminalOnce() {
    if (!terminalSessionId) return;
    const response = await fetch(`/api/terminal/poll?session_id=${encodeURIComponent(terminalSessionId)}&cursor=${terminalCursor}`);
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.error || "终端轮询失败");

    if (data.chunk) appendTerminalOutput(data.chunk);
    terminalCursor = Number(data.cursor || terminalCursor || 0);
    if (data.done) {
      clearTerminalPoll();
      const code = typeof data.exit_code === "number" ? data.exit_code : null;
      if (code === 0) setTerminalState("Exited (0)", "accepted");
      else setTerminalState(`Exited (${code === null ? "?" : code})`, code === null ? "waiting" : "warning");
      terminalSessionId = "";
    }
  }

  function startTerminalPolling() {
    clearTerminalPoll();
    terminalPollTimer = window.setInterval(async () => {
      try {
        await pollTerminalOnce();
      } catch (error) {
        clearTerminalPoll();
        setTerminalState("Terminal Error", "error");
        showError(error.message || String(error));
      }
    }, 350);
  }

  async function startTerminal(language) {
    const lang = language === "shell" ? "shell" : currentLanguage;
    const payload = {
      language: lang,
      runtime_env: runtimeEnvSelect.value || "system",
      cwd: currentDir,
    };
    if (lang !== "shell") {
      const source = getEditorValue();
      if (!source.trim()) throw new Error("请先输入代码");
      if (source.length > limits.maxCodeChars) throw new Error(`代码长度超过 ${limits.maxCodeChars}`);
      payload.code = source;
    }

    terminalStartBtn.disabled = true;
    shellStartBtn.disabled = true;
    clearTerminalOutput();
    setTerminalState("Starting...", "waiting");
    try {
      const response = await fetch("/api/terminal/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok || !data.ok) throw new Error(data.error || "启动终端失败");

      terminalSessionId = String(data.session_id || "");
      terminalCursor = 0;
      setTerminalState(`Running ${lang}`, "accepted");
      await pollTerminalOnce();
      startTerminalPolling();
    } finally {
      terminalStartBtn.disabled = false;
      shellStartBtn.disabled = false;
    }
  }

  async function stopTerminal() {
    clearTerminalPoll();
    const response = await fetch("/api/terminal/stop", { method: "POST" });
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.error || "停止终端失败");
    terminalSessionId = "";
    setTerminalState("Stopped", "warning");
  }

  async function sendTerminalInput() {
    const line = terminalInputEl.value;
    if (!line.trim()) return;
    if (!terminalSessionId) throw new Error("终端未启动");

    const response = await fetch("/api/terminal/input", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: terminalSessionId, data: `${line}\n` }),
    });
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.error || "发送输入失败");
    terminalInputEl.value = "";
  }

  async function runCode() {
    const source = getEditorValue();
    if (!source.trim()) throw new Error("请先输入代码");
    if (source.length > limits.maxCodeChars) throw new Error(`代码长度超过 ${limits.maxCodeChars}`);
    if (stdinBox.value.length > limits.maxStdinChars) throw new Error(`stdin 长度超过 ${limits.maxStdinChars}`);

    setVerdict("Running");
    stdoutView.textContent = "正在运行...";
    clearError();
    runBtn.disabled = true;
    try {
      const response = await fetch("/run-code", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          language: currentLanguage,
          code: source,
          stdin: stdinBox.value || "",
          expected_output: expectedBox.value || "",
          runtime_env: runtimeEnvSelect.value || "system",
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.stderr || data.error || "运行失败");

      setVerdict(data.verdict || (data.ok ? "Accepted" : "Error"));
      stdoutView.textContent = data.stdout || "(no output)";
      if (data.stderr) showError(data.stderr);
      else clearError();
    } finally {
      runBtn.disabled = false;
    }
  }

  function fmtBytes(value) {
    const n = Number(value || 0);
    if (!Number.isFinite(n) || n <= 0) return "--";
    const units = ["B", "KB", "MB", "GB", "TB"];
    let num = n;
    let idx = 0;
    while (num >= 1024 && idx < units.length - 1) {
      num /= 1024;
      idx += 1;
    }
    return `${num >= 10 ? num.toFixed(1) : num.toFixed(2)} ${units[idx]}`;
  }

  function fmtPercent(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return "--";
    return `${n.toFixed(1)}%`;
  }

  async function refreshServerStatus() {
    refreshStatusBtn.disabled = true;
    statusErrorEl.textContent = "";
    try {
      const response = await fetch("/api/server-status");
      const data = await response.json();
      if (!response.ok || !data.ok) throw new Error(data.error || "读取服务器状态失败");

      const cpu = Number((data.cpu && data.cpu.usage_percent) || data.cpu_percent || 0);
      const mem = data.memory || {};
      const disk = data.disk || {};

      cpuLoadEl.textContent = fmtPercent(cpu);
      memUsedEl.textContent = mem.used_human || fmtBytes(mem.used_bytes);
      memTotalEl.textContent = mem.total_human || fmtBytes(mem.total_bytes);
      diskUsedEl.textContent = disk.used_human || fmtBytes(disk.used_bytes);
      diskTotalEl.textContent = disk.total_human || fmtBytes(disk.total_bytes);
      diskPathEl.textContent = String(disk.path || "/");
      statusTimeEl.textContent = `更新于 ${new Date().toLocaleTimeString()}`;
    } catch (error) {
      statusErrorEl.textContent = error.message || String(error);
    } finally {
      refreshStatusBtn.disabled = false;
    }
  }

  function toggleSearchPanel(force) {
    const show = typeof force === "boolean" ? force : searchPanelEl.classList.contains("is-hidden");
    searchPanelEl.classList.toggle("is-hidden", !show);
    if (show) {
      gitPanelEl.classList.add("is-hidden");
      searchInputEl.focus();
      searchInputEl.select();
    }
  }

  function toggleGitPanel(force) {
    const show = typeof force === "boolean" ? force : gitPanelEl.classList.contains("is-hidden");
    gitPanelEl.classList.toggle("is-hidden", !show);
    if (show) {
      searchPanelEl.classList.add("is-hidden");
      refreshGitStatus().catch((error) => showError(error.message || String(error)));
    }
  }

  const commands = [
    { id: "save", title: "Save Current File", run: () => saveCurrentTab() },
    { id: "run", title: "Run Code", run: () => runCode() },
    { id: "quick-open", title: "Quick Open", run: () => openQuickOpen() },
    { id: "search", title: "Global Search", run: () => toggleSearchPanel(true) },
    { id: "source-control", title: "Refresh Source Control", run: () => refreshGitStatus() },
    { id: "new-file", title: "New File", run: () => createItem("file") },
    { id: "new-folder", title: "New Folder", run: () => createItem("folder") },
    { id: "terminal-code", title: "Start Code Terminal", run: () => startTerminal(currentLanguage) },
    { id: "terminal-shell", title: "Start Shell Terminal", run: () => startTerminal("shell") },
  ];

  function renderCommandList(items) {
    commandListEl.replaceChildren();
    if (!items.length) {
      const empty = document.createElement("li");
      empty.className = "search-empty";
      empty.textContent = "无命令匹配";
      commandListEl.appendChild(empty);
      return;
    }
    for (const item of items) {
      const li = document.createElement("li");
      li.className = "search-result-item";
      li.innerHTML = `<button type="button" class="search-result-btn" data-command="${item.id}"><span>${item.title}</span><small>${item.id}</small></button>`;
      commandListEl.appendChild(li);
    }
    commandListEl.querySelectorAll("button[data-command]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        closeCommandPalette();
        const id = btn.dataset.command || "";
        const command = commands.find((item) => item.id === id);
        if (command) await command.run();
      });
    });
  }

  function openCommandPalette() {
    commandOverlay.classList.remove("is-hidden");
    commandInputEl.value = "";
    renderCommandList(commands);
    commandInputEl.focus();
  }

  function closeCommandPalette() {
    commandOverlay.classList.add("is-hidden");
  }

  function openQuickOpen() {
    quickOpenOverlay.classList.remove("is-hidden");
    quickOpenInputEl.value = "";
    quickOpenListEl.replaceChildren();
    quickOpenInputEl.focus();
  }

  function closeQuickOpen() {
    quickOpenOverlay.classList.add("is-hidden");
  }

  function bindEvents() {
    langTabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        const lang = tab.dataset.lang === "cpp" ? "cpp" : "python";
        setLanguage(lang);
        const active = getTab(activeTabPath);
        if (active) active.language = lang;
      });
    });

    runtimeEnvSelect.addEventListener("change", () => setVerdict("Env changed"));
    refreshEnvBtn.addEventListener("click", () => refreshRuntimeEnvs().catch((error) => showError(error.message || String(error))));
    runBtn.addEventListener("click", () => runCode().catch((error) => showError(error.message || String(error))));
    saveBtn.addEventListener("click", () => saveCurrentTab().catch((error) => showError(error.message || String(error))));
    newFileBtn.addEventListener("click", () => createItem("file").catch((error) => showError(error.message || String(error))));
    newFolderBtn.addEventListener("click", () => createItem("folder").catch((error) => showError(error.message || String(error))));
    renameItemBtn.addEventListener("click", () => renamePath("").catch((error) => showError(error.message || String(error))));
    deleteItemBtn.addEventListener("click", () => deletePath("").catch((error) => showError(error.message || String(error))));

    terminalStartBtn.addEventListener("click", () => startTerminal(currentLanguage).catch((error) => showError(error.message || String(error))));
    shellStartBtn.addEventListener("click", () => startTerminal("shell").catch((error) => showError(error.message || String(error))));
    terminalStopBtn.addEventListener("click", () => stopTerminal().catch((error) => showError(error.message || String(error))));
    terminalSendBtn.addEventListener("click", () => sendTerminalInput().catch((error) => showError(error.message || String(error))));
    terminalInputEl.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        sendTerminalInput().catch((error) => showError(error.message || String(error)));
      }
    });

    openDirBtn.addEventListener("click", () => {
      const value = cwdInput.value.trim();
      loadDirectory(value).catch((error) => showError(error.message || String(error)));
    });
    upDirBtn.addEventListener("click", () => {
      if (!parentDir) return;
      loadDirectory(parentDir).catch((error) => showError(error.message || String(error)));
    });
    homeDirBtn.addEventListener("click", () => loadDirectory(homeDir).catch((error) => showError(error.message || String(error))));

    fileListEl.addEventListener("click", (event) => {
      const btn = event.target instanceof HTMLElement ? event.target.closest("button[data-action]") : null;
      if (!btn) return;
      const action = btn.getAttribute("data-action") || "";
      const pathText = btn.getAttribute("data-path") || "";
      const kind = btn.getAttribute("data-kind") || "file";

      if (action === "open") {
        selectedPath = pathText;
        if (kind === "dir") loadDirectory(pathText).catch((error) => showError(error.message || String(error)));
        else {
          const tab = getTab(pathText);
          if (tab) switchTab(pathText, true);
          else openFile(pathText).catch((error) => showError(error.message || String(error)));
        }
      } else if (action === "rename") {
        renamePath(pathText).catch((error) => showError(error.message || String(error)));
      } else if (action === "delete") {
        deletePath(pathText).catch((error) => showError(error.message || String(error)));
      }
    });

    editorTabsEl.addEventListener("click", (event) => {
      const target = event.target;
      if (!(target instanceof HTMLElement)) return;
      if (target.dataset.action === "close") {
        closeTab(target.dataset.path || "");
        return;
      }
      const tabBtn = target.closest(".editor-tab");
      if (!tabBtn) return;
      const pathText = tabBtn.getAttribute("data-path") || "";
      if (pathText) switchTab(pathText, true);
    });

    searchBtnEl.addEventListener("click", () => runGlobalSearch().catch((error) => showError(error.message || String(error))));
    searchInputEl.addEventListener("keydown", (event) => {
      if (event.key === "Enter") runGlobalSearch().catch((error) => showError(error.message || String(error)));
    });

    gitRefreshBtn.addEventListener("click", () => refreshGitStatus().catch((error) => showError(error.message || String(error))));
    gitCommitBtn.addEventListener("click", () => commitGitChanges().catch((error) => showError(error.message || String(error))));

    toggleCommandBtn.addEventListener("click", openCommandPalette);
    toggleQuickOpenBtn.addEventListener("click", openQuickOpen);
    toggleSearchBtn.addEventListener("click", () => toggleSearchPanel());
    toggleGitBtn.addEventListener("click", () => toggleGitPanel());

    commandInputEl.addEventListener("input", () => {
      const q = commandInputEl.value.trim().toLowerCase();
      if (!q) {
        renderCommandList(commands);
        return;
      }
      renderCommandList(commands.filter((item) => item.id.includes(q) || item.title.toLowerCase().includes(q)));
    });
    commandInputEl.addEventListener("keydown", async (event) => {
      if (event.key === "Escape") {
        closeCommandPalette();
      } else if (event.key === "Enter") {
        const first = commandListEl.querySelector("button[data-command]");
        if (first instanceof HTMLButtonElement) first.click();
      }
    });

    quickOpenInputEl.addEventListener("input", () => runQuickOpenSearch().catch((error) => showError(error.message || String(error))));
    quickOpenInputEl.addEventListener("keydown", (event) => {
      if (event.key === "Escape") closeQuickOpen();
      if (event.key === "Enter") {
        const first = quickOpenListEl.querySelector("button[data-path]");
        if (first instanceof HTMLButtonElement) first.click();
      }
    });

    commandOverlay.addEventListener("click", (event) => {
      if (event.target === commandOverlay) closeCommandPalette();
    });
    quickOpenOverlay.addEventListener("click", (event) => {
      if (event.target === quickOpenOverlay) closeQuickOpen();
    });

    refreshStatusBtn.addEventListener("click", () => refreshServerStatus().catch((error) => showError(error.message || String(error))));

    document.addEventListener("keydown", (event) => {
      if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key.toLowerCase() === "p") {
        event.preventDefault();
        openCommandPalette();
        return;
      }
      if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key.toLowerCase() === "f") {
        event.preventDefault();
        toggleSearchPanel(true);
        return;
      }
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
        event.preventDefault();
        saveCurrentTab().catch((error) => showError(error.message || String(error)));
        return;
      }
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "p") {
        event.preventDefault();
        openQuickOpen();
        return;
      }
      if (event.key === "F9") {
        event.preventDefault();
        runCode().catch((error) => showError(error.message || String(error)));
        return;
      }
      if (event.key === "Escape") {
        closeCommandPalette();
        closeQuickOpen();
      }
    });
  }

  async function initMonaco() {
    if (!window.require || !window.monaco) {
      setupFallbackEditor();
      return;
    }

    await new Promise((resolve) => {
      window.require.config({ paths: { vs: "https://cdn.jsdelivr.net/npm/monaco-editor@0.52.2/min/vs" } });
      window.require(["vs/editor/editor.main"], () => resolve(null));
    });

    monacoEditor = window.monaco.editor.create(codeEditorHost, {
      value: "",
      language: currentLanguage,
      theme: "vs-dark",
      automaticLayout: true,
      minimap: { enabled: true },
      fontSize: 14,
      tabSize: 2,
      insertSpaces: true,
      smoothScrolling: true,
      scrollBeyondLastLine: false,
    });
    usingMonaco = true;
    codeEditorFallback.style.display = "none";
    codeEditorHost.style.display = "block";
  }

  async function bootstrap() {
    try {
      setLanguage("python");
      applyRuntimeEnvs(boot.runtimeEnvs || [], boot.currentRuntimeEnv || "system");
      bindEvents();
      await initMonaco();
      await loadDirectory(currentDir || boot.initialDir || "");
      setVerdict("Waiting");
      setTerminalState("Idle", "waiting");

      refreshServerStatus().catch(() => null);
      statusTimer = window.setInterval(() => {
        refreshServerStatus().catch(() => null);
      }, 3000);
    } catch (error) {
      showError(error.message || String(error));
      setVerdict("Error");
    }
  }

  window.addEventListener("beforeunload", () => {
    clearTerminalPoll();
    if (statusTimer) {
      window.clearInterval(statusTimer);
      statusTimer = null;
    }
  });

  bootstrap();
})();
