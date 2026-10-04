/**
 * PM2 process file for OpenType miner training.
 *
 *   cd /workspace/opentype-miner
 *   pm2 start ecosystem.config.cjs --only ot-train-reads
 *   pm2 logs ot-train-reads
 *
 * Secrets: put HF_TOKEN in ./.env (gitignored). The train wrapper sources it.
 * One GPU: do not run reads + harness together.
 */
const fs = require("fs");
const path = require("path");
const root = __dirname;

const envPath = path.join(root, ".env");
if (fs.existsSync(envPath)) {
  for (const line of fs.readFileSync(envPath, "utf8").split("\n")) {
    const m = line.match(/^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$/);
    if (!m) continue;
    let v = m[2].trim();
    if (
      (v.startsWith('"') && v.endsWith('"')) ||
      (v.startsWith("'") && v.endsWith("'"))
    ) {
      v = v.slice(1, -1);
    }
    if (process.env[m[1]] === undefined) process.env[m[1]] = v;
  }
}

const championRepo = process.env.CHAMPION_REPO || "ebobo/m-0e98bd7f";
const championRev =
  process.env.CHAMPION_REV || "5824c0e82d99edb1b1aafc2ba424f564bd277c72";

module.exports = {
  apps: [
    {
      name: "ot-train-reads",
      cwd: root,
      script: path.join(root, "scripts", "run_train_reads.sh"),
      interpreter: "bash",
      autorestart: false,
      max_restarts: 0,
      kill_timeout: 60000,
      env: {
        PYTHONUNBUFFERED: "1",
        HF_HOME: process.env.HF_HOME || "/workspace/.hf_home",
        CUDA_VISIBLE_DEVICES: process.env.CUDA_VISIBLE_DEVICES || "0",
        CHAMPION_REPO: championRepo,
        CHAMPION_REV: championRev,
        READ_MAX_STEPS: process.env.READ_MAX_STEPS || "2000",
      },
      out_file: path.join(root, "results", "pm2-reads.out.log"),
      error_file: path.join(root, "results", "pm2-reads.err.log"),
      merge_logs: true,
      time: true,
    },
    {
      name: "ot-train-harness",
      cwd: root,
      script: path.join(root, "scripts", "run_train_harness.sh"),
      interpreter: "bash",
      autorestart: false,
      max_restarts: 0,
      kill_timeout: 60000,
      env: {
        PYTHONUNBUFFERED: "1",
        HF_HOME: process.env.HF_HOME || "/workspace/.hf_home",
        CUDA_VISIBLE_DEVICES: process.env.CUDA_VISIBLE_DEVICES || "0",
        CHAMPION_REPO: championRepo,
        CHAMPION_REV: championRev,
        HARNESS_MAX_STEPS: process.env.HARNESS_MAX_STEPS || "2000",
      },
      out_file: path.join(root, "results", "pm2-harness.out.log"),
      error_file: path.join(root, "results", "pm2-harness.err.log"),
      merge_logs: true,
      time: true,
    },
  ],
};
