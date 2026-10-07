import json
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

API_KEY = os.getenv("TECHHELPER_API_KEY", "").strip()

llm = ChatOpenAI(
    base_url=os.getenv("TECHHELPER_BASE_URL", "https://api.deepseek.com/v1"),
    api_key=API_KEY,
    model=os.getenv("TECHHELPER_MODEL", "deepseek-chat"),
)

MAX_ROUNDS = 6

#表单模板
class ClarifyDecision(BaseModel):
    needs_more: bool = Field(description="是否还要继续提问")
    next_question: str = Field(default="", description="下一轮要问的问题，每次只问一个")
    goal: str = Field(default="", description="用户最终目标 X")
    attempt: str = Field(default="", description="用户当前尝试的方案 Y")
    phenomenon: str = Field(default="", description="客观现象")
    environment: str = Field(default="", description="运行环境")
    error_info: str = Field(default="", description="报错/日志")
    steps: str = Field(default="", description="复现步骤")
    expected: str = Field(default="", description="期望结果")
    actual: str = Field(default="", description="实际结果")
    constraints: str = Field(default="", description="约束")
    xy_suspected: bool = Field(default=False, description="是否疑似 XY 问题")
    xy_reason: str = Field(default="", description="疑似 XY 问题的理由")

FIELD_KEYS = [
    "goal",
    "attempt",
    "phenomenon",
    "environment",
    "error_info",
    "steps",
    "expected",
    "actual",
    "constraints",
]

CLARIFY_SYSTEM = """\
你是技术提问澄清助手 TechHelper。用户是非技术人员，你要通过多轮提问帮他整理出一份规范的技术求助工单。
规则：
1. 每次只问一个问题，要短、说人话。
2. 区分【用户最终目标 X】和【用户当前尝试的方案 Y】。
   若用户说"帮我做/装/改 XX"，要追问背后真正想解决的异常现象或目标。
3. 只把客观现象当事实，主观猜想记为 Y。
4. 已收集的信息要保留；本轮只填写新获得的信息，没获得就留空字符串。
5. 目标、现象、环境、复现、期望、实际这些基本齐全时，及时结束（needs_more=false）。
"""

# 提示词模板
clarify_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CLARIFY_SYSTEM),
        ("system", "当前已收集的信息（JSON）：{fields_json}"),
        ("system", "当前是第 {round} 轮，最多 {max_rounds} 轮。"),
        MessagesPlaceholder("history"),
    ]
)
#链
clarify_chain = clarify_prompt | llm.with_structured_output(ClarifyDecision)


#工单模板
def render_ticket(fields: dict, xy_suspected: bool, xy_reason: str, round_no: int) -> str:
    def v(key: str) -> str:
        return (fields.get(key) or "").strip() or "（未提供）"

    xy = f"疑似 XY 问题：{xy_reason}" if (xy_suspected and xy_reason) else "未发现明显 XY 问题"
    return f"""# 技术求助工单
> 澄清轮次：{round_no}

- 目标(X)：{v('goal')}
- 当前尝试(Y)：{v('attempt')}
- 现象：{v('phenomenon')}
- 环境：{v('environment')}
- 报错/日志：{v('error_info')}
- 复现步骤：{v('steps')}
- 期望结果：{v('expected')}
- 实际结果：{v('actual')}
- 约束：{v('constraints')}
- 疑似XY问题：{xy}
"""


#工具
@tool
def get_log_instructions(environment: str) -> str:
    env = (environment or "").lower()
    if "windows" in env:
        return (
            "Windows 获取日志：\n"
            "1) 按 Win+R 输入 eventvwr.msc 打开事件查看器；\n"
            "2) 报错弹窗按 Ctrl+C 可复制文字；\n"
            "3) PowerShell 只读命令：\n"
            "   Get-WinEvent -LogName Application -MaxEvents 20 | Where-Object LevelDisplayName -eq '错误'"
        )
    if "mac" in env or "macos" in env or "osx" in env:
        return (
            "macOS 获取日志：\n"
            "1) 打开「控制台」(Console) 应用；\n"
            "2) 终端只读命令：log show --last 30m --style compact"
        )
    return (
        "Linux 获取日志：\n"
        "1) 应用日志：tail -n 100 /var/log/应用名.log；\n"
        "2) 系统日志：journalctl -xe --no-pager | tail -n 100"
    )


@tool
def make_helper_script(task: str, environment: str) -> str:
    """根据任务和环境，生成一个只读、安全、带注释的辅助脚本。"""
    env = (environment or "").lower()
    if "windows" in env:
        return (
            "# PowerShell 辅助脚本（只读，不修改系统）\n"
            f"# 任务：{task}\n"
            "# 请逐行阅读确认后再执行\n"
            "Write-Output '环境信息:'\n"
            "Get-ComputerInfo | Select-Object OsName, OsVersion\n"
            "Write-Output '最近应用错误:'\n"
            "Get-WinEvent -LogName Application -MaxEvents 20 | Where-Object LevelDisplayName -eq '错误'"
        )
    return (
        "#!/usr/bin/env bash\n"
        f"# 任务：{task}\n"
        "# 只读脚本，请确认后再执行\n"
        "echo '== 系统信息 =='\n"
        "uname -a\n"
        "echo '== 最近系统日志 =='\n"
        "journalctl -xe --no-pager | tail -n 50 || dmesg | tail -n 50"
    )


#Agent（自动循环调用工具）
FOLLOWUP_SYSTEM = """\
你是 TechHelper 的专家追问转译助手。你会收到一份技术求助工单 + 技术专家的追问（可能含术语）。
请：1) 用大白话解释专家到底想让他提供什么；2) 解释术语；
3) 需要取日志/数据时，调用 get_log_instructions 工具，按用户环境给出步骤。"""

SOLUTION_SYSTEM = """\
你是 TechHelper 的方案落地助手。你会收到技术求助工单 + 专家方案。
请：1) 把方案转成非技术用户能照做的一步步操作（编号）；
2) 需要自动化时调用 make_helper_script 工具生成只读脚本；
3) 明确风险点，绝不建议破坏性/不可逆操作。"""

followup_agent = create_agent(llm, tools=[get_log_instructions], system_prompt=FOLLOWUP_SYSTEM)
solution_agent = create_agent(llm, tools=[make_helper_script], system_prompt=SOLUTION_SYSTEM)


def run_agent(agent, text: str) -> str:
    result = agent.invoke({"messages": [HumanMessage(content=text)]})
    return result["messages"][-1].content


#主流程
def main() -> None:
    fields: dict = {}
    history: list = []  # 记忆：只存用户和助手的对话
    round_no = 0
    xy_suspected = False
    xy_reason = ""
    ticket = ""
    while True:
        user = input("你：").strip()
        if user in ("exit", "退出", "q"):
            break
        #澄清阶段
        if not ticket:
            history.append(HumanMessage(content=user))
            decision = clarify_chain.invoke(
                {
                    "history": history,
                    "fields_json": json.dumps(fields, ensure_ascii=False),
                    "round": round_no + 1,
                    "max_rounds": MAX_ROUNDS,
                }
            )
            for k in FIELD_KEYS:
                v = getattr(decision, k, "")
                if v:
                    fields[k] = v
            if decision.xy_suspected:
                xy_suspected = True
                if decision.xy_reason:
                    xy_reason = decision.xy_reason
            round_no += 1

            if decision.needs_more and round_no < MAX_ROUNDS:
                history.append(AIMessage(content=decision.next_question))
                print(f"\n助手：{decision.next_question}")
            else:
                ticket = render_ticket(fields, xy_suspected, xy_reason, round_no)
                print(f"\n助手：信息收集完毕，工单如下：\n{ticket}")
                print("（可把工单复制给专家；拿到追问或方案后，用 /追问 或 /方案 贴给我）")
            continue

        #追问/方案阶段（Agent）
        if user.startswith("/追问"):
            q = user[len("/追问"):].strip()
            if not q:
                print("用法：/追问 专家的追问内容")
                continue
            print(f"\n助手：{run_agent(followup_agent, f'工单：\n{ticket}\n\n专家追问：{q}')}")
        elif user.startswith("/方案"):
            s = user[len("/方案"):].strip()
            if not s:
                print("用法：/方案 专家的方案内容")
                continue
            print(f"\n助手：{run_agent(solution_agent, f'工单：\n{ticket}\n\n专家方案：{s}')}")
        else:
            print("工单已生成。可输入 /追问 或 /方案，或 exit 退出。")


if __name__ == "__main__":
    main()
