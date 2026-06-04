
import os
import re
import textwrap
import time
import random
from rich import print
from typing import List, Optional, Dict, Any

from langchain.chat_models import AzureChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.callbacks import get_openai_callback, OpenAICallbackHandler

from rap.scenario.envScenario import EnvScenario
from openai import OpenAI


delimiter = "####"

example_message = textwrap.dedent(f"""\
        {delimiter} Driving scenario description:
        You are driving on a road with 4 lanes, and you are currently driving in the second lane from the left. Your speed is 25.00 m/s, acceleration is 0.00 m/s^2, and lane position is 363.14 m. 
        There are other vehicles driving around you, and below is their basic information:
        - Vehicle `912` is driving on the same lane of you and is ahead of you. The speed of it is 23.30 m/s, acceleration is 0.00 m/s^2, and lane position is 382.33 m.
        - Vehicle `864` is driving on the lane to your right and is ahead of you. The speed of it is 21.30 m/s, acceleration is 0.00 m/s^2, and lane position is 373.74 m.
        - Vehicle `488` is driving on the lane to your left and is ahead of you. The speed of it is 23.61 $m/s$, acceleration is 0.00 $m/s^2$, and lane position is 368.75 $m$.

        {delimiter} Your available actions:
        IDLE - remain in the current lane with current speed Action_id: 1
        Turn-left - change lane to the left of the current lane Action_id: 0
        Turn-right - change lane to the right of the current lane Action_id: 2
        Acceleration - accelerate the vehicle Action_id: 3
        Deceleration - decelerate the vehicle Action_id: 4
        """)

example_answer = textwrap.dedent(f"""\
        Well, I have 5 actions to choose from. Now, I would like to know which action is possible. 
        I should first check if I can acceleration, then idle, finally decelerate.  I can also try to change lanes but with caution and not too frequently.

        - I want to know if I can accelerate, so I need to observe the car in front of me on the current lane, which is car `912`. The distance between me and car `912` is 382.33 - 363.14 = 19.19 m, and the difference in speed is 23.30 - 25.00 = -1.7 m/s. Car `912` is traveling 19.19 m ahead of me and its speed is 1.7 m/s slower than mine. This distance is too close and my speed is too high, so I should not accelerate.
        - Since I cannot accelerate, I want to know if I can maintain my current speed. I need to observe the car in front of me on the current lane, which is car `912`. The distance between me and car `912` is 382.33 - 363.14 = 19.19 m, and the difference in speed is 23.30 - 25.00 = -1.7 m/s. Car `912` is traveling 19.19 m ahead of me and its speed is 1.7 m/s slower than mine. This distance is too close and my speed is too high, so if I maintain my current speed, I may collide with it.
        - Maintain my current speed is not a good idea, so I can only decelearate to keep me safe on my current lane. Deceleraion is a feasible action.
        - Besides decelearation, I can also try to change lanes. I should carefully check the distance and speed of the cars in front of me on the left and right lanes. Noted that change-lane is not a frequent action, so I should not change lanes too frequently.
        - I first try to change lanes to the left. The car in front of me on the left lane is car `488`. The distance between me and car `488` is 368.75-363.14=5.61 m, and the difference in speed is 23.61 - 25.00=-1.39 m/s. Car `488` is traveling 5.61 m ahead of me and its speed is 1.39 m/s slower than mine. This distance is too close, the safety lane-change distance is 25m. Besides, my speed is higher than the front car on the left lane. If I change lane to the left, I may collide with it.                                           So I cannot change lanes to the left.
        - Now I want to see if I can change lanes to the right. The car in front of me on the right lane is car 864. The distance between me and car 864 is 373.74-363.14 = 10.6 m, and the difference in speed is 23.61-25.00=-3.7 m/s. Car 864 is traveling 10.6 m ahead of me and its speed is 3.7 m/s slower than mine. The distance is too close and my speed is higher than the front car on the right lane. the safety lane-change distance is 25m. if I change lanes to the right, I may collide with it. So I cannot change lanes to the right.
        - Now my only option is to slow down to keep me safe.
        Final Answer: Deceleration
                                         
        Response to user:#### 4
        """)


def _fmt_ttc(x: Optional[float]) -> str:
    if x is None:
        return "N/A"
    if x == float("inf"):
        return "inf"
    try:
        return f"{float(x):.2f}"
    except Exception:
        return str(x)


class DriverAgent:
    def __init__(
        self,
        sce: EnvScenario,
        temperature: float = 0,
        verbose: bool = False,
    ) -> None:
        self.sce = sce
        self.temperature = temperature
        self.verbose = verbose
        self.oai_api_type = os.getenv("OPENAI_API_TYPE")

        if self.oai_api_type == "azure":
            print("Using Azure Chat API")
            self.llm = AzureChatOpenAI(
                callbacks=[OpenAICallbackHandler()],
                deployment_name=os.getenv("AZURE_CHAT_DEPLOY_NAME"),
                temperature=temperature,
                max_tokens=2000,
                request_timeout=60,
                streaming=True,
            )
            self.client = None
            self.model_name = None

        elif self.oai_api_type == "openai":
            print("Use OpenAI API via zhizengzeng (OpenAI SDK)")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is empty. Please set it.")

            self.client = OpenAI(
                api_key=api_key,
                base_url=os.getenv("OPENAI_API_BASE", "https://api.zhizengzeng.com/v1"),
            )

            self.model_name = os.getenv("OPENAI_CHAT_MODEL")
            if not self.model_name:
                raise ValueError("OPENAI_CHAT_MODEL is empty. Please set the model name.")

            if self.verbose:
                print(f"[dim]OpenAI base_url={os.getenv('OPENAI_API_BASE', 'https://api.zhizengzeng.com/v1')} "
                      f"model={self.model_name}[/dim]")

            self.llm = None

        else:
            raise ValueError(f"Unknown OPENAI_API_TYPE: {self.oai_api_type}")

    # ---------------------------
    # Robust OpenAI helpers
    # ---------------------------
    def _extract_content_from_completion(self, completion) -> str:
        """
        1) 若上游返回了 error（即使HTTP看起来成功），优先抛出明确错误
        2) 否则再从 choices[0].message.content 提取
        """
        if completion is None:
            raise RuntimeError("OpenAI completion is None (upstream returned null response).")

        err = getattr(completion, "error", None)
        if err:
            if isinstance(err, dict):
                code = err.get("code")
                etype = err.get("type")
                msg = err.get("message")
            else:
                code = getattr(err, "code", None)
                etype = getattr(err, "type", None)
                msg = getattr(err, "message", str(err))

            raise RuntimeError(f"Upstream returned error: code={code}, type={etype}, message={msg}")

        choices = getattr(completion, "choices", None)
        if not choices:
            dumped = None
            try:
                dumped = completion.model_dump()
            except Exception:
                dumped = str(completion)

            if isinstance(dumped, dict) and dumped.get("error"):
                e = dumped["error"]
                raise RuntimeError(
                    f"Upstream returned error in dump: code={e.get('code')}, "
                    f"type={e.get('type')}, message={e.get('message')}"
                )

            raise RuntimeError(f"OpenAI completion has no choices. Raw completion dump: {dumped}")

        msg = getattr(choices[0], "message", None)
        if msg is None:
            raise RuntimeError("OpenAI completion.choices[0].message is None.")

        content = getattr(msg, "content", None)
        if content is None:
            raise RuntimeError("OpenAI completion.choices[0].message.content is None.")

        return content

    def _call_openai_chat_with_retry(
        self,
        chat_messages: List[Dict[str, Any]],
        max_tokens: int = 2000,
        temperature: float = 0.0,
        max_retries: int = 4,
        base_sleep: float = 0.8,
    ) -> str:
        last_err: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=chat_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return self._extract_content_from_completion(completion)

    def _extract_action_id(self, text: str) -> Optional[int]:
        if not text:
            return None
        m = re.findall(r"\b([0-4])\b", text)
        if not m:
            return None
        return int(m[-1])

    # ---------------------------
    # Prompt builder
    # ---------------------------
    def _build_system_message(
        self,
        driving_intensions: str,
        ttc_threshold: float,
        ttc_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        base_intro = f"""
        You are ChatGPT, a large language model trained by OpenAI. Now you act as a mature driving assistant, who can give accurate and correct advice for human driver in complex urban driving scenarios.
        You will be given a detailed description of the driving scenario of current frame along with your history of previous decisions. You will also be given the available actions you are allowed to take. All of these elements are delimited by {delimiter}.

        Your response should use the following format:
        <reasoning>
        <reasoning>
        <repeat until you have a decision>
        Response to user:{delimiter} <only output one `Action_id` as a int number of you decision, without any action name or explanation. The output decision must be unique and not ambiguous, for example if you decide to decelearate, then output `4`> 

        Make sure to include {delimiter} to separate every step.
        """

        if not ttc_info:
            return textwrap.dedent(base_intro)

        keep_ttc = ttc_info.get("keep", {}).get("ttc", float("inf"))
        left_info = ttc_info.get("left", None)
        right_info = ttc_info.get("right", None)
        left_ttc = left_info.get("ttc", float("inf")) if left_info else float("inf")
        right_ttc = right_info.get("ttc", float("inf")) if right_info else float("inf")
        min_all = ttc_info.get("ttc_min_all", float("inf"))

        if min_all is None or min_all == float("inf") or float(min_all) >= ttc_threshold:
            return textwrap.dedent(base_intro)

        print("ttc_min_all < ttc_threshold,(SAFETY OVERRIDE MODE)")

        ban_left = (left_ttc != float("inf")) and (float(left_ttc) < ttc_threshold)
        ban_right = (right_ttc != float("inf")) and (float(right_ttc) < ttc_threshold)

        ban_lines = []
        if ban_left:
            ban_lines.append("- Do NOT choose Turn-left (Action 0) because the left lane is unsafe (low TTC).")
        if ban_right:
            ban_lines.append("- Do NOT choose Turn-right (Action 2) because the right lane is unsafe (low TTC).")
        if not ban_lines:
            ban_lines.append("- Avoid lane changes (Action 0 / 2) unless they are clearly safer than staying and decelerating.")

        safety_extra = f"""
        [SAFETY OVERRIDE MODE]

        Estimated TTC summary (seconds):
        - keep lane TTC: {_fmt_ttc(keep_ttc)}
        - left lane TTC: {_fmt_ttc(left_ttc)}
        - right lane TTC: {_fmt_ttc(right_ttc)}
        - min TTC overall: {_fmt_ttc(min_all)}
        Safety threshold: {ttc_threshold:.2f} seconds.

        In this mode, you MUST:
        - Prioritize safety and collision avoidance over efficiency or comfort.
        - NEVER choose acceleration (Action 3) when min TTC is below the threshold.
        - Strongly prefer deceleration (Action 4) to reduce collision risk.
        {chr(10).join(ban_lines)}
        - If multiple actions seem reasonable, always choose the MOST conservative and safest action.

        Additional guidance:
        - If keep-lane TTC is low, deceleration is the default safe action.
        - Only change lane if the target lane TTC is clearly higher AND does not introduce rear-collision risk.
        """

        return textwrap.dedent(base_intro + safety_extra)

    # ---------------------------
    # Main decision function
    # ---------------------------
    def few_shot_decision(
        self,
        scenario_description: str = "Not available",
        previous_decisions: str = "Not available",
        available_actions: str = "Not available",
        driving_intensions: str = "Not available",
        fewshot_messages: List[str] = None,
        fewshot_answers: List[str] = None,
        ttc_info: Optional[Dict[str, Any]] = None,
        ttc_threshold: float = 3.0,
        traj_pred_text: Optional[str] = None,  
    ):
        system_message = self._build_system_message(
            driving_intensions=driving_intensions,
            ttc_threshold=ttc_threshold,
            ttc_info=ttc_info,
        )

        human_message = f"""\
        Above messages are some examples of how you make a decision successfully in the past. Those scenarios are similar to the current scenario. You should refer to those examples to make a decision for the current scenario. 

        Here is the current scenario:
        {delimiter} Driving scenario description:
        {scenario_description}
        {delimiter} Driving Intensions:
        {driving_intensions}
        {delimiter} Available actions:
        {available_actions}
        """

        if ttc_info is not None:
            keep_ttc = ttc_info.get("keep", {}).get("ttc", float("inf"))
            left_info = ttc_info.get("left", None)
            right_info = ttc_info.get("right", None)
            left_ttc = left_info.get("ttc", float("inf")) if left_info else float("inf")
            right_ttc = right_info.get("ttc", float("inf")) if right_info else float("inf")
            min_all = ttc_info.get("ttc_min_all", float("inf"))

            human_message += f"""
        {delimiter} TTC Safety Summary (seconds):
        keep lane TTC: {_fmt_ttc(keep_ttc)}
        left lane TTC: {_fmt_ttc(left_ttc)}
        right lane TTC: {_fmt_ttc(right_ttc)}
        min TTC overall: {_fmt_ttc(min_all)}
        Safety threshold: {ttc_threshold:.2f}

        Interpretation:
        - Smaller TTC => higher collision risk.
        - If a target lane TTC is below the threshold, that lane-change is unsafe (rear or front collision risk).
        """

        human_message += """
        You can stop reasoning once you have a valid action to take. 
        """
        human_message = human_message.replace("        ", "")

        if traj_pred_text:
            human_message += traj_pred_text
            print(f"all prompt{human_message}")

        if fewshot_messages is None or fewshot_answers is None:
            raise ValueError("fewshot_messages or fewshot_answers is None")

        messages = [SystemMessage(content=system_message)]
        
        for i in range(len(fewshot_messages)):
            messages.append(HumanMessage(content=fewshot_messages[i]))
            messages.append(AIMessage(content=fewshot_answers[i]))
        messages.append(HumanMessage(content=human_message))
        print("[cyan]Agent answer:[/cyan]")
        response_content = ""

        if self.oai_api_type == "azure":
            for chunk in self.llm.stream(messages):
                response_content += chunk.content
                print(chunk.content, end="", flush=True)
            print("\n")

        elif self.oai_api_type == "openai":
            chat_messages: List[Dict[str, Any]] = []
            for m in messages:
                if isinstance(m, SystemMessage):
                    role = "system"
                elif isinstance(m, HumanMessage):
                    role = "user"
                elif isinstance(m, AIMessage):
                    role = "assistant"
                else:
                    continue
                chat_messages.append({"role": role, "content": m.content})

            response_content = self._call_openai_chat_with_retry(
                chat_messages=chat_messages,
                temperature=self.temperature,
                max_tokens=2000,
                max_retries=4,
                base_sleep=0.8,
            )
            print(response_content)
            print("\n")

        else:
            raise ValueError(f"Unknown OPENAI_API_TYPE: {self.oai_api_type}")

        result = self._extract_action_id(response_content)

        if result is None:
            print("Output is not a valid action_id (0-4), checking the output...")

            check_message = f"""
            You are a output checking assistant who is responsible for checking the output of another agent.
            
            The output you received is: {response_content}

            Your should just output the right int type of action_id, with no other characters or delimiters.
            i.e. :
            | Action_id | Action Description                                     |
            |--------|--------------------------------------------------------|
            | 0      | Turn-left: change lane to the left of the current lane |
            | 1      | IDLE: remain in the current lane with current speed   |
            | 2      | Turn-right: change lane to the right of the current lane|
            | 3      | Acceleration: accelerate the vehicle                 |
            | 4      | Deceleration: decelerate the vehicle                 |

            You answer format would be:
            {delimiter} <correct action_id within 0-4>
            """

            if self.oai_api_type == "azure":
                messages_check = [HumanMessage(content=check_message)]
                with get_openai_callback() as cb:
                    check_response = self.llm(messages_check)
                check_content = check_response.content

            elif self.oai_api_type == "openai":
                check_content = self._call_openai_chat_with_retry(
                    chat_messages=[{"role": "user", "content": check_message}],
                    temperature=0.0,
                    max_tokens=50,
                    max_retries=4,
                    base_sleep=0.6,
                )
            else:
                raise ValueError(f"Unknown OPENAI_API_TYPE: {self.oai_api_type}")

            checked = self._extract_action_id(check_content)
            if checked is None:
                raise RuntimeError(f"Output check failed. Raw checker output: {check_content}")
            result = checked

        few_shot_answers_store = ""
        for i in range(len(fewshot_messages)):
            few_shot_answers_store += fewshot_answers[i] + "\n---------------\n"

        print("Result:", result)
        return result, response_content, human_message, few_shot_answers_store
