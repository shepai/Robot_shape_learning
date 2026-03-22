
from ollama import chat
import requests
import sys 
class Decisions:
    def __init__(self,model="gemma3"):
        self.MODEL=model
        self.history = []
        self.system_prompt = [
            """You are a robot arm tasked with manipulating objects. Follow these instructions carefully:

1. **Movement**: Use `move_gripper_to(target_coord)` to move the gripper hand to a specific coordinate. The coordinate must be given as a 3D point in the form `(x, y, z)`. Ensure that the gripper hovers over the object with a vertical offset of 0.15 along the z-axis.

2. **Picking**: To pick up a block, use `pick_block(index)` where `index` is the identifier of the block you want to pick.

3. **Placing**: To place the block, use `put_block()`.

4. **Task Generation**: For any given task, generate a sequence of commands to solve it. You are **only allowed** to use the three commands above (no explanations or other actions).
"""," ",
"""
**Format**: 
- Use the exact syntax of the commands.
- If parameters are involved, make sure to list them correctly.
- Your response should consist of only the generated commands in the following format:
    - `pick_block(index)`
    - `move_gripper_to(x, y, z)`
    - `put_block()`"""
        ]
    def chat(self,reading):
        for i in range(len(self.system_prompt)-1):
            messages = self.system_prompt[i]+"\n"

        messages += f"Current environment: {reading}\n "+self.system_prompt[-1]

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.MODEL,
                    "prompt": messages,
                    "stream": False
                }
            )
            response.raise_for_status()
        except Exception as e:
            print("Error communicating with Ollama:", e, file=sys.stderr)
            return ""

        reply = response.json().get("response", "")

        # Save turn to history
        #self.history.append((reading, reply))
        return reply
    def set_task(self,task):
        self.system_prompt[1]="TASK: "+task+". Only use the commands move_gripper_to(target_coord), pick_block(index), put_block()"

if __name__=="__main__":
    dec_model=Decisions()
    dec_model.set_task("You are given two blocks and you need to stack them on top of eachother")
    reply=dec_model.chat("Block 1 at (0,0.3,0), block 2 at (0.2,0.3,0)")
    print(reply)

    