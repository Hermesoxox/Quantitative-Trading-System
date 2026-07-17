#!/usr/bin/env bash
# ============================================================================
# 在 Mac 上一键安装并试跑本量化策略（适合零基础）。
#
# 用法（在「终端」里）：
#   1) 用 cd 进入项目文件夹，例如：
#        cd ~/Desktop/Quantitative-Trading-System
#   2) 运行：
#        bash run_on_mac.sh
#
# 脚本会：创建独立运行环境(.venv) -> 安装依赖 -> 跑自检 -> 跑一次小规模真实回测。
# 之后再用第 6 节命令跑完整策略或盘后信号即可。
# ============================================================================
set -e

# 境内 Mac 默认优先用东财/腾讯数据源(更快、更稳)；海外用户可改成 auto。
export QTS_SOURCE="${QTS_SOURCE:-cn}"

echo "==> 1/5 检查 Python3"
if ! command -v python3 >/dev/null 2>&1; then
  echo "❌ 没找到 python3。请先到 https://www.python.org/downloads/ 下载安装后重试。"
  exit 1
fi
python3 --version

echo "==> 2/5 创建独立运行环境 .venv（只需一次）"
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> 3/5 安装依赖（首次较慢，请耐心等待）"
python -m pip install --upgrade pip >/dev/null
# 国内网络可加清华镜像：在下面命令末尾加  -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install -r requirements.txt

# lightgbm 在 Mac 上需要 OpenMP 运行库；缺了会自动回退，但装上更好
if ! python -c "import lightgbm" >/dev/null 2>&1; then
  echo "ℹ️ 提示：如想启用 LightGBM（更强选股），可执行： brew install libomp"
  echo "   不装也行——程序会自动回退到等效的线性/sklearn 模型，不影响运行。"
fi

echo "==> 4/5 运行自检（冒烟测试，应显示『所有冒烟测试通过』）"
python tests/test_smoke.py

echo "==> 5/5 跑一次小规模真实数据回测（20 只蓝筹，验证数据连通）"
python -m examples.run_advanced --real --n 20 || {
  echo "⚠️ 真实数据回测失败，多半是数据接口暂时不通。"
  echo "   可改用模拟数据先体验： python -m examples.run_backtest"
  exit 0
}

cat <<'TIP'

✅ 全部完成！

★★ 重要：每次新开终端窗口，必须先执行下面这一行激活环境 ★★
（执行后命令行开头出现 (.venv) 字样才算成功，否则会报 command not found: python）

  source .venv/bin/activate

★★ 复制命令时：以 # 开头的说明行不要复制，只复制命令本身 ★★

常用命令：

  python -m examples.run_advanced --real --n 40
      完整真实回测（≤5 只集中持仓，约 40 只蓝筹池）

  python -m examples.run_walkforward --real --n 40
      最诚实的纯样本外回测

  python -m examples.live_signal --capital 1000000
      盘后实盘信号（每天收盘后跑一次，给“明日买/卖清单”）

  python -m examples.run_advanced --real --codes all --n 300
      全市场大股票池验证（首次约 1-2 小时建缓存，之后快）

结果图保存在 reports_advanced/ 、reports_walkforward/ 等文件夹里。
提示：海外网络环境可改用 Yahoo 源： export QTS_SOURCE=auto
TIP
