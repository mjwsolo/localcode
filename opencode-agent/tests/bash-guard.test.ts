const src = await Bun.file(new URL("../plugins/localcode.ts", import.meta.url).pathname).text();
const m = src.match(/const PROG = ([\s\S]*?);\nconst SERVER_CMD = new RegExp\(([\s\S]*?)\n\);/);
if (!m) throw new Error("regex block not found");
const PROG = eval(m[1]); const SERVER_CMD = eval(`new RegExp(${m[2]})`);
const block = ["npm run dev", "npx vite", "vite", "vite --port 5173", "cd app && npm start", "python3 -m http.server 8000", "uvicorn app:app", "npm run preview"];
const allow = ["npm create vite@latest tmp -- --template react-ts", "npm install -D vite @vitejs/plugin-react vite-plugin-pwa", "sleep 15 && tail -5 /tmp/x.log && ls node_modules/.bin/ | grep -E \"vite|tailwind\"", "npx vite build", "npm run build", "vite build", "npm test", "echo vite", "npx tsc --noEmit"];
let bad = 0;
for (const c of block) if (!SERVER_CMD.test(c)) { console.log("MISSED block:", c); bad++; }
for (const c of allow) if (SERVER_CMD.test(c)) { console.log("FALSE block:", c); bad++; }
if (bad) { console.error(`guard has ${bad} problems`); process.exit(1); } console.log("guard OK");
