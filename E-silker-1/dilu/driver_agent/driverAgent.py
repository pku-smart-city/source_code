import os
import sys
import textwrap
import time
import random
from typing import List

from rich import print
import re
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI, ChatOllama
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.callbacks import get_openai_callback, OpenAICallbackHandler, StreamingStdOutCallbackHandler
from collections import Counter
from dilu.scenario.envScenario import EnvScenario


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


class DriverAgent:
    def __init__(
        self, sce: EnvScenario,
        temperature: float = 0, verbose: bool = False
    ) -> None:
        self.sce = sce
        oai_api_type = os.getenv("OPENAI_API_TYPE")
        if oai_api_type == "azure":
            print("Using Azure Chat API")
            self.llm = AzureChatOpenAI(
                callbacks=[
                    OpenAICallbackHandler()
                ],
                deployment_name=os.getenv("AZURE_CHAT_DEPLOY_NAME"),
                temperature=temperature,
                max_tokens=2000,
                request_timeout=60,
                streaming=True,
            )
        elif oai_api_type == "openai":
            print("Use OpenAI API")
            self.llm = ChatOpenAI(
                temperature=temperature,
                callbacks=[
                    OpenAICallbackHandler()
                ],
                model_name=os.getenv("OPENAI_CHAT_MODEL"),
                max_tokens=2000,
                request_timeout=60,
                streaming=True,
            )
        elif oai_api_type == "deepseek-R1-32B":
            print("Using deepseek-R1 32B")
            self.llm = ChatOllama(model="deepseek-r1:32b")  # or any other model in ollama

    def few_shot_decision(self, scenario_description: str = "Not available", previous_decisions: str = "Not available",
                          available_actions: str = "Not available", driving_intensions: str = "Not available",
                          fewshot_messages: List[str] = None, fewshot_answers: List[str] = None):
        # for template usage refer to: https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/

        system_message = textwrap.dedent(f"""\
        You are ChatGPT, a large language model trained by OpenAI. Now you act as a mature driving assistant, who can give accurate and correct advice for human driver in complex urban driving scenarios.
        You will be given a detailed description of the driving scenario of current frame along with your history of previous decisions. You will also be given the available actions you are allowed to take. All of these elements are delimited by {delimiter}.

        Your response should use the following format:
        <reasoning>
        <reasoning>
        <repeat until you have a decision>
        Response to user:{delimiter} <only output one `Action_id` as a int number of you decision, without any action name or explanation. The output decision must be unique and not ambiguous, for example if you decide to decelearate, then output `4`> 

        Make sure to include {delimiter} to separate every step.
        """)

        human_message = f"""\
        Above messages are some examples of how you make a decision successfully in the past. Those scenarios are similar to the current scenario. You should refer to those examples to make a decision for the current scenario. 

        Here is the current scenario:
        {delimiter} Driving scenario description:
        {scenario_description}
        {delimiter} Driving Intensions:
        {driving_intensions}
        {delimiter} Available actions:
        {available_actions}

        You can stop reasoning once you have a valid action to take. 
        """
        human_message = human_message.replace("        ", "")

        if fewshot_messages is None:
            raise ValueError("fewshot_message is None")
        messages = [
            SystemMessage(content=system_message),
            # HumanMessage(content=example_message),
            # AIMessage(content=example_answer),
        ]
        for i in range(len(fewshot_messages)):
            messages.append(
                HumanMessage(content=fewshot_messages[i])
            )
            messages.append(
                AIMessage(content=fewshot_answers[i])
            )
        messages.append(
            HumanMessage(content=human_message)
        )
        # print("fewshot number:", (len(messages) - 2)/2)
        start_time = time.time()
        # with get_openai_callback() as cb:
        # response = self.llm(messages)
        # print(response.content)
        print("[cyan]Agent answer:[/cyan]")
        response_content = ""

        for chunk in self.llm.stream(messages):
            response_content += chunk.content
            print(chunk.content, end="", flush=True)
        print("\n")
        decision_action = response_content.split(delimiter)[-1]
        try:
            result = int(decision_action)
            if result < 0 or result > 4:
                raise ValueError
        except ValueError:
            print("Output is not a int number, checking the output...")
            check_message = f"""
            You are a output checking assistant who is responsible for checking the output of another agent.

            The output you received is: {decision_action}

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
            messages = [
                HumanMessage(content=check_message),
            ]
            with get_openai_callback() as cb:
                check_response = self.llm(messages)
            result = int(check_response.content.split(delimiter)[-1])

        few_shot_answers_store = ""
        for i in range(len(fewshot_messages)):
            few_shot_answers_store += fewshot_answers[i] + \
                                      "\n---------------\n"
        print("Result:", result)
        return result, response_content, human_message, few_shot_answers_store


    def few_shot_decision_V1(self, scenario_description: str = "Not available", previous_decisions: str = "Not available", available_actions: str = "Not available", driving_intensions: str = "Not available", fewshot_messages: List[str] = None, fewshot_answers: List[str] = None, frame_id = None,real_data = None):
        # for template usage refer to: https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/

        system_message = textwrap.dedent(f"""\
        You are DeepSeek-R1, a large language model trained by DeepSeek. Now you act as a mature driving assistant, who can give accurate and correct advice for human driver in complex urban driving scenarios.
        
    
        
        You are tasked with making decisions for vehicles in a traffic simulation scenario. The goal is to ensure that the global speed trend of all vehicles in the simulation matches the real-world data provided. 
        
        The real-world data consists of 60 entries, each representing the normalized speeds of all vehicles on the road at each second. The speeds are normalized using the mean and standard deviation calculated from all speed values across all time steps.

        The speed values are binned into the following intervals: [-∞, -5.5, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, ∞]. For each time step, the proportion of speeds falling into each bin is used to describe the speed distribution at that moment.
        
        
        
        You will be given a detailed description of the driving scenario of current frame along with your history of previous decisions. You will also be given the available actions you are allowed to take. Besides, you will be given the current decision frame number and the speed distribution of real-world data as well. All of these elements are delimited by {delimiter}.

        Your response should use the following format:
        <reasoning>
        <reasoning>
        <repeat until you have a decision>
        Response to user:{delimiter} <only output one `Action_id` as a int number of you decision, without any action name or explanation. The output decision must be unique and not ambiguous, for example if you decide to decelearate, then output `4`> 

        Make sure to include {delimiter} to separate every step.
        """)

        human_message = f"""\
        Above messages are some examples of how you make a decision successfully in the past. Those scenarios are similar to the current scenario. You should refer to those examples to make a decision for the current scenario. 

        Here is the current scenario:
        {delimiter} Driving scenario description:
        {scenario_description}
        {delimiter} Current decision frame number:
        {frame_id}
        {delimiter} Driving Intensions:
        {driving_intensions}
        {delimiter} Available actions:
        {available_actions}
        {delimiter} Speed distribution of real-world data:
        {real_data}
        
        
        Decision Constraints At each frame t (1 ≤ t ≤ 60):
            a. Use the normalized Driving scenario description data to infer the global traffic conditions.
            b. Adjust your speed to steer the simulation-wide speed distribution toward the real-world histogram for frame t.
            c. Prioritize matching the proportion of vehicles in critical bins (e.g., bins near 0.0 or high-traffic zones).
        
        You need to make decisions according to the following process:
        1. you need to develop a plan to guide subsequent action. The following is the process for generating the plan:
                Step 1:Brainstorm all workable and distinct plans based on the current scenario. Here are some example of the plan's content: "decelerate and then merge behind Vehicle '87'" , "merge ahead of Vehicle '87'".
                Step 2:For each of the proposed plans, evaluate their potential. Consider their pros and cons, implementation difficulty, potential challenges and Changes in global traffic conditions caused by it. Assign safety, efficiency score from 0 to 10 and similarity score with Real-world data to each option based on these factors. 
                Step 3:Based on the evaluations and scenarios, rank the plans.
                Step 4:Choose one plan as your driving plan according to your own idea.
        2.Analyse all the available actions and then make reasonable action choices based on the current scene information and the plan. If the plan was to slow down or speed up first and then change lanes, you need to firstly analyze whether you can make the lane change to complete the plan now. The most important thing is that the next decision is 1s later, so you need to ensure that the state after the decision is executed for 1s is safe. 
        3.Attentions: Each step needs you to reasoning. Changing lanes into the lane of a vehicle which parallel to you can cause a collision! Do not change lanes left and right frequently.


        You can stop reasoning once you have a valid action to take. 
        """
        human_message = human_message.replace("        ", "")

        if fewshot_messages is None:
            raise ValueError("fewshot_message is None")
        messages = [
            SystemMessage(content=system_message),
            # HumanMessage(content=example_message),
            # AIMessage(content=example_answer),
        ]
        for i in range(len(fewshot_messages)):
            messages.append(
                HumanMessage(content=fewshot_messages[i])
            )
            messages.append(
                AIMessage(content=fewshot_answers[i])
            )
        messages.append(
            HumanMessage(content=human_message)
        )
        # print("fewshot number:", (len(messages) - 2)/2)
        start_time = time.time()
        # with get_openai_callback() as cb:
        # response = self.llm(messages)
        # print(response.content)
        print("[cyan]Agent answer:[/cyan]")
        response_content = ""

        # print("--------------------------------shuru-------------------------")
        # print(messages)
        # print("--------------------------------shuru-------------------------")
        #
        # sys.exit()
        for chunk in self.llm.stream(messages):
            response_content += chunk.content
            print(chunk.content, end="", flush=True)
        print("\n")
        decision_action = response_content.split(delimiter)[-1]
        # try:
        #     result = int(decision_action)
        #     if result < 0 or result > 4:
        #         raise ValueError
        # except ValueError:
        #     print("Output is not a int number, checking the output...")
        #     check_message = f"""
        #     You are a output checking assistant who is responsible for checking the output of another agent.
        #
        #     The output you received is: {decision_action}
        #
        #     Your should just output the right int type of action_id, with no other characters or delimiters.
        #     i.e. :
        #     | Action_id | Action Description                                     |
        #     |--------|--------------------------------------------------------|
        #     | 0      | Turn-left: change lane to the left of the current lane |
        #     | 1      | IDLE: remain in the current lane with current speed   |
        #     | 2      | Turn-right: change lane to the right of the current lane|
        #     | 3      | Acceleration: accelerate the vehicle                 |
        #     | 4      | Deceleration: decelerate the vehicle                 |
        #
        #
        #     You answer format would be:
        #     {delimiter} <correct action_id within 0-4>
        #     """
        #     messages = [
        #         HumanMessage(content=check_message),
        #     ]
        #     with get_openai_callback() as cb:
        #         check_response = self.llm(messages)
        #     result = int(check_response.content.split(delimiter)[-1])
        #
        # few_shot_answers_store = ""
        # for i in range(len(fewshot_messages)):
        #     few_shot_answers_store += fewshot_answers[i] + \
        #         "\n---------------\n"
        # print("Result:", result)
        # return result, response_content, human_message, few_shot_answers_store
        for i in range(5):
            try:
                result = int(decision_action)
                if result < 0 or result > 4:
                    raise ValueError
                else:
                    break
            except:
                if i == 4:
                    check_message = f""" Please only return a number between 0-4 
                                             You answer format would be:
                                             {delimiter} <correct action_id within 0-4>"""
                    messages = [
                            HumanMessage(content=check_message),
                        ]
                    with get_openai_callback() as cb:
                        check_response = self.llm(messages)
                    result = re.findall(r'[0-4]', check_response)[-1]

                else:
                    print("Output is not a int number, checking the output...")
                    check_message = f"""
                                                   You are a output checking assistant who is responsible for checking the output of another agent.

                                                   The output you received is: {decision_action}

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
                    messages = [
                        HumanMessage(content=check_message),
                    ]
                    with get_openai_callback() as cb:
                        check_response = self.llm(messages)
                    decision_action = check_response.content.split(delimiter)[-1]

        print("Result:", result)
        few_shot_answers_store = ""
        for i in range(len(fewshot_messages)):
            few_shot_answers_store += fewshot_answers[i] + \
                                          "\n---------------\n"

        return result, response_content, human_message, few_shot_answers_store


    def few_shot_decision_V2(self, scenario_description: str = "Not available", previous_decisions: str = "Not available",
                             available_actions: str = "Not available", driving_intensions: str = "Not available",
                             fewshot_messages: List[str] = None, fewshot_answers: List[str] = None, frame_id = None,real_data = None):
        # for template usage refer to: https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/

        system_message = textwrap.dedent(f"""\
        You are DeepSeek-R1, a large language model trained by DeepSeek. Now you act as a mature driving assistant, who can give accurate and correct advice for human driver in complex urban driving scenarios.
        You will be given a detailed description of the driving scenario of current frame along with your history of previous decisions. 
        You will also be given the available actions you are allowed to take. 
        Additionally, you will get the current driving scenario description and real-world data.
        All of these elements are delimited by {delimiter}.

        Your response should use the following format:
        <reasoning>
        <reasoning>
        <repeat until you have a decision>
        Response to user:{delimiter} <only output one `Action_id` as a int number of you decision, without any action name or explanation. The output decision must be unique and not ambiguous, for example if you decide to decelearate, then output `4`> 

        Make sure to include {delimiter} to separate every step.
        """)

        human_message = f"""\
        Above messages are some examples of how you make a decision successfully in the past. Those scenarios are similar to the current scenario. You should refer to those examples to make a decision for the current scenario. 

        Here is the current scenario:
        {delimiter} Driving scenario description:
        {scenario_description}
        {delimiter} Real-world data:
        {real_data}
        {delimiter} Available actions:
        {available_actions}
        {delimiter} Driving Intensions:
        {driving_intensions}
        
        
       

        You can stop reasoning once you have a valid action to take. 
        """

        # print('------------------------------------------------------')
        # print(scenario_description)
        # print('------------------------------------------------------')
        # print(real_data)
        # print('------------------------------------------------------')
        # print(available_actions)
        # print('------------------------------------------------------')
        # print(driving_intensions)

        # {delimiter}
        # Current
        # decision
        # frame
        # id:
        # {frame_id}

        human_message = human_message.replace("        ", "")

        if fewshot_messages is None:
            raise ValueError("fewshot_message is None")
        messages = [
            SystemMessage(content=system_message),
            # HumanMessage(content=example_message),
            # AIMessage(content=example_answer),
        ]
        for i in range(len(fewshot_messages)):
            messages.append(
                HumanMessage(content=fewshot_messages[i])
            )
            messages.append(
                AIMessage(content=fewshot_answers[i])
            )
        messages.append(
            HumanMessage(content=human_message)
        )

        print("[cyan]Agent answer:[/cyan]")
        response_content = ""

        for chunk in self.llm.stream(messages):
            response_content += chunk.content
            print(chunk.content, end="", flush=True)
        print("\n")
        decision_action = response_content.split(delimiter)[-1]

        try:
            result = int(decision_action)
            if result < 0 or result > 4:
                raise ValueError
        except ValueError:
            print("Output is not a int number, checking the output...")
            check_message = f"""
                    You are a output checking assistant who is responsible for checking the output of another agent.

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

            messages = [
                AIMessage(content=response_content),
                HumanMessage(content=check_message)
            ]
            with get_openai_callback() as cb:
                check_response = self.llm(messages)

            try:
                print(check_response.content)
            except Exception:
                import sys
                sys.stdout.write(check_response.content + "\n")

            result = re.findall(r'[0-4]', check_response.content)[-1]

            response_content += f'{delimiter} Response to user:{delimiter} {result}'

        few_shot_answers_store = ""
        for i in range(len(fewshot_messages)):
            few_shot_answers_store += fewshot_answers[i] + \
                                      "\n---------------\n"
        print("Result:", result)

        return result, response_content, human_message, few_shot_answers_store

    def prompt_llm(self,
                   scenario_description: str = "Not available",
                   available_actions: str = "Not available",
                   driving_intensions: str = "Not available",
                   real_data=None):
        system_message = textwrap.dedent(f"""\
                You are DeepSeek-R1, a large language model trained by DeepSeek. Now you act as a mature driving assistant, who can give accurate and correct advice for human driver in complex urban driving scenarios.
                You will be given a detailed description of the driving scenario of current frame along with your history of previous decisions. 
                You will also be given the available actions you are allowed to take. 
                Additionally, you will get the current driving scenario description and real-world data.
                All of these elements are delimited by {delimiter}.
                Response to user:{delimiter} <only output one `Action_id` as a int number of you decision, without any action name or explanation. The output decision must be unique and not ambiguous, for example if you decide to decelearate, then output `4`> 
                Make sure to include {delimiter} to separate every step.
                """)

        human_message = f"""\
                Above messages are some examples of how you make a decision successfully in the past. Those scenarios are similar to the current scenario. You should refer to those examples to make a decision for the current scenario. 

                Here is the current scenario:
                {delimiter} Driving scenario description:
                {scenario_description}
                {delimiter} Real-world data:
                {real_data}
                {delimiter} Available actions:
                {available_actions}
                {delimiter} Driving Intensions:
                {driving_intensions}
                
                You can stop reasoning once you have a valid action to take. 
                """

        human_message = human_message.replace("        ", "")

        messages = [
            SystemMessage(content=system_message),
        ]

        messages.append(
            HumanMessage(content=human_message)
        )

        print("[cyan]Agent answer:[/cyan]")
        response_content = ""

        for chunk in self.llm.stream(messages):
            response_content += chunk.content
            print(chunk.content, end="", flush=True)
        print("\n")
        decision_action = response_content.split(delimiter)[-1]

        try:
            result = int(decision_action)
            if result < 0 or result > 4:
                raise ValueError
        except ValueError:
            print("Output is not a int number, checking the output...")
            check_message = f"""
                            You are a output checking assistant who is responsible for checking the output of another agent.

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

            messages = [
                AIMessage(content=response_content),
                HumanMessage(content=check_message)
            ]
            with get_openai_callback() as cb:
                check_response = self.llm(messages)

            try:
                print(check_response.content)
            except Exception:
                import sys
                sys.stdout.write(check_response.content + "\n")

            result = re.findall(r'[0-4]', check_response.content)[-1]

            response_content += f'{delimiter} Response to user:{delimiter} {result}'

        # print("Result:", result)

        return result



    def few_shot_decision_V3(self, scenario_description: str = "Not available",
                             previous_decisions: str = "Not available",
                             available_actions: str = "Not available", driving_intensions: str = "Not available",
                             fewshot_messages: List[str] = None, fewshot_answers: List[str] = None, frame_id=None,
                             real_data=None):
        # for template usage refer to: https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/



        prompts = [
            '''
                Make a driving decision based on the following requirements:
    
                Consistency with Real Data: Based on the <<Driving scenario description>>, match a specific vehicle from the <<Real-world data>> that best fits the current driving scenario. Adjust your driving style to mimic the matched vehicle’s behavior as closely as possible.
                
                Additional Instructions:
                
                When matching a vehicle from the real-world data, consider factors such as road layout, traffic conditions, and the positions and speeds of surrounding vehicles to ensure the match is contextually appropriate.
                Ensure that your driving style aligns as closely as possible with the behavior of the matched vehicle.
                Provide a brief explanation of your decision, including which vehicle was matched and how its behavior influenced your choice.
            ''',
            '''
                Make a driving decision based on the following requirements:

                Consistency with Real Data: Based on the <<Driving scenario description>>, match a specific vehicle from the <<Real-world data>> that best fits the current driving scenario.Adjust your driving style to mimic the matched vehicle’s behavior as closely as possible.However, you may increase your speed (up to the speed limit) and perform lane changes and improve driving efficiency, even if the matched vehicle did not take those actions.
                Future State: Since the next decision will occur after {self.detectN} frames, your current decision must consider state after {self.detectN} frames.

                Additional Instructions:

                When matching a vehicle from the real-world data, consider factors such as road layout, traffic conditions, and the positions and speeds of surrounding vehicles to ensure the match is contextually appropriate.
                Ensure that the chosen driving style aligns with the behavior of the matched vehicle.
                Provide a brief explanation of your decision, including which vehicle was matched, how its behavior influenced your choice, and whether you made any adjustments for efficiency.
            '''
            ,
            '''
                Make a driving decision based on the following requirements:
    
                Safety First: Your decision must keep the vehicle safe, avoiding collisions for at least the next {self.detectN} frames.
                Consistency with Real Data: Based on the <<Driving scenario description>>, match a specific vehicle from the <<Real-world data>> that best fits the current driving scenario.
                Adjust your driving style to mimic the matched vehicle’s behavior as closely as possible, while ensuring safety.
                However, you may increase your speed (up to the speed limit) and perform lane changes if they can be done safely and improve driving efficiency, even if the matched vehicle did not take those actions.
                Future State: Since the next decision will occur after {self.detectN} frames, your current decision must lead to a safe state after {self.detectN} frames.
    
                Additional Instructions:
    
                When matching a vehicle from the real-world data, consider factors such as road layout, traffic conditions, and the positions and speeds of surrounding vehicles to ensure the match is contextually appropriate.
                Ensure that the chosen driving style aligns with the behavior of the matched vehicle while maintaining safety and avoiding collisions. If you make adjustments for efficiency, ensure they do not compromise safety.
                Provide a brief explanation of your decision, including which vehicle was matched, how its behavior influenced your choice, and whether you made any adjustments for efficiency.
            ''',




        #
        #           '''
        #                           Make a driving decision based on the following requirements:
        #
        #                           Consistency with Real Data: Based on the <<Driving scenario description>>, match a specific vehicle from the <<Real-world data>> that best fits the current driving scenario. Adjust your driving style to mimic the matched vehicle’s behavior as closely as possible.
        #
        #                           Additional Instructions:
        #
        #                           When matching a vehicle from the real-world data, consider factors such as road layout, traffic conditions, and the positions and speeds of surrounding vehicles to ensure the match is contextually appropriate.
        #                           Ensure that your driving style aligns as closely as possible with the behavior of the matched vehicle.
        #                           Provide a brief explanation of your decision, including which vehicle was matched and how its behavior influenced your choice.
        #                       ''',
        # '''
        #     Make a driving decision based on the following requirements:
        #
        #     Safety First: Your decision must keep the vehicle safe, avoiding collisions for at least the next {self.detectN} frames.
        #     Consistency with Real Data: Based on the <<Driving scenario description>>, match a specific vehicle from the <<Real-world data>> that best fits the current driving scenario.
        #     Adjust your driving style to mimic the matched vehicle’s behavior as closely as possible, while ensuring safety.
        #     However, you may increase your speed (up to the speed limit) and perform lane changes if they can be done safely and improve driving efficiency, even if the matched vehicle did not take those actions.
        #     Future State: Since the next decision will occur after {self.detectN} frames, your current decision must lead to a safe state after {self.detectN} frames.
        #
        #     Additional Instructions:
        #
        #     When matching a vehicle from the real-world data, consider factors such as road layout, traffic conditions, and the positions and speeds of surrounding vehicles to ensure the match is contextually appropriate.
        #     Ensure that the chosen driving style aligns with the behavior of the matched vehicle while maintaining safety and avoiding collisions. If you make adjustments for efficiency, ensure they do not compromise safety.
        #     Provide a brief explanation of your decision, including which vehicle was matched, how its behavior influenced your choice, and whether you made any adjustments for efficiency.
        # '''
        ]

        results = []

        for i, p in enumerate(prompts):
            print(f"Decision :{i}")
            results.append(self.prompt_llm(scenario_description, available_actions, p, real_data))

        # 统计元素频率
        counter = Counter(results)

        # 找出最高频率值
        max_count = max(counter.values())

        # 收集所有达到最高频率的元素
        most_frequent_items = [item for item, count in counter.items() if count == max_count]

        # 随机选择一个最高频元素
        if max_count != 1:
            result = random.choice(most_frequent_items)
        else:
            result = results[2]

        print("Result:", result)

        return result, None, None, None





