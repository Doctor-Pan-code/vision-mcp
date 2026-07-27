#!/usr/bin/env node

/**
 * Vision MCP Server - Node.js 包装入口
 * ======================================
 * 让 npx vision-mcp 能直接运行 Python MCP 服务器。
 * 
 * 工作方式：
 *   1. 定位 server.py（相对于此脚本的位置）
 *   2. 用 child_process.spawn 启动 Python
 *   3. 透传所有命令行参数、环境变量和 stdio
 *   4. 退出码也透传
 * 
 * 使用示例：
 *   npx vision-mcp                          # stdio 模式（默认）
 *   npx vision-mcp --transport sse --port 8000  # SSE 模式
 *   npx vision-mcp --help                   # 查看帮助
 */

const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");
const process = require("process");

// 获取 server.py 的路径（与 bin/ 同级）
const scriptDir = path.resolve(__dirname, "..");
const serverPy = path.join(scriptDir, "server.py");

// 检查 server.py 是否存在
if (!fs.existsSync(serverPy)) {
  console.error("[vision-mcp] 错误: 找不到 server.py");
  console.error("  预期路径: %s", serverPy);
  console.error("  请确保 vision-mcp 包安装完整。");
  process.exit(1);
}

// 检测 Python 命令（试 python3 再试 python）
const pythonCmd = (() => {
  if (process.platform === "win32") return "python";
  // macOS/Linux: 优先 python3，fallback python
  try {
    require("child_process").execSync("python3 --version", { stdio: "ignore" });
    return "python3";
  } catch {
    return "python";
  }
})();

// 启动 Python 进程
const child = spawn(pythonCmd, [serverPy, ...process.argv.slice(2)], {
  stdio: ["inherit", "inherit", "inherit"],
  env: { ...process.env },
  windowsHide: false,
});

// 透传退出码
child.on("exit", (code) => {
  process.exit(code ?? 0);
});

child.on("error", (err) => {
  console.error("[vision-mcp] 启动 Python 失败:", err.message);
  console.error("  请确保已安装 Python 3.10+ (https://python.org)");
  console.error("  如果已安装 Python，请检查是否安装了必要依赖：");
  console.error("    pip install -r %s", path.join(scriptDir, "requirements.txt"));
  process.exit(1);
});
