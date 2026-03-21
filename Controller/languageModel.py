
from ollama import chat
import requests
import sys 
class Decisions:
    def __init__(self,model="gemma3"):
        self.MODEL=model
        self.history = []
        self.system_prompt = [
            "You are a robot arm which can pick up objects with its tractor beam on the hand. You just need to hover the gripper over the z with an offset of 0.15 where the coordinates are in the form (x,y,z) otherwise you crush the object",
            "Your only actions you can do are move_gripper_to(target_coord) to move the gripper hand to a specific coordinate, pick_block(index) to pick up a specific block, put_block() to drop whatever is held. Given a task generate the steps to solving it using only these listed commands",
            " "
        ]
    def chat(self,reading):
        for i in range(len(self.system_prompt)):
            messages = self.system_prompt[i]+"\n"
        """for user, ai in self.history:
            messages += f"User input: {user}\nAI: {ai}\n" """

        messages += f"User input: {reading}\nAI: "

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
        self.system_prompt[2]="TASK: "+task+". Only use the commands move_gripper_to(target_coord), pick_block(index), put_block()"

if __name__=="__main__":
    dec_model=Decisions()
    dec_model.set_task("You are given two blocks and you need to stack them on top of eachother")
    reply=dec_model.chat("Block 1 at (0,0.3,0), block 2 at (0.2,0.3,0)")
    print(reply)

    