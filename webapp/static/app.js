const STARTER_QUESTIONS = [
  "Кто такой Махмуд Кашгари?",
  "Какую книгу написал Юсуф Баласагуни?",
  "Что такое Кутадгу билиг?",
  "Кто такой Ходжа Ахмед Яссауи?",
  "В каком веке жил Юсуф Баласагуни?",
];

const messagesEl = document.getElementById("messages");
const emptyStateEl = document.getElementById("empty-state");
const composerEl = document.getElementById("composer");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send");
const chipsEl = document.getElementById("chips");

function renderChips() {
  for (const q of STARTER_QUESTIONS) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.textContent = q;
    chip.addEventListener("click", () => {
      inputEl.value = q;
      ask(q);
    });
    chipsEl.appendChild(chip);
  }
}

function addMessage(role, text) {
  if (emptyStateEl) emptyStateEl.remove();
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.textContent = text;
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return el;
}

function addTyping() {
  const el = document.createElement("div");
  el.className = "typing";
  el.innerHTML = "<span></span><span></span><span></span>";
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return el;
}

async function ask(question) {
  const trimmed = question.trim();
  if (!trimmed) return;

  addMessage("user", trimmed);
  inputEl.value = "";
  inputEl.disabled = true;
  sendBtn.disabled = true;

  const typingEl = addTyping();

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: trimmed }),
    });

    const data = await res.json().catch(() => null);
    typingEl.remove();

    if (!res.ok) {
      const detail = data && data.detail ? data.detail : `Ошибка сервера (${res.status})`;
      addMessage("error", detail);
      return;
    }

    addMessage("bot", data.answer);
  } catch (err) {
    typingEl.remove();
    addMessage("error", "Не удалось связаться с сервером. Проверь, что server.py запущен.");
  } finally {
    inputEl.disabled = false;
    sendBtn.disabled = false;
    inputEl.focus();
  }
}

composerEl.addEventListener("submit", (e) => {
  e.preventDefault();
  ask(inputEl.value);
});

renderChips();
inputEl.focus();
