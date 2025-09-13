from transformers import pipeline
from langchain_core.prompts import PromptTemplate

class PromptManager:
    def __init__(self):
        self.prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum and keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.

        <context>
        {context}
        </context>

        Question: {question}

        Answer:"""

    def create_prompt(self) -> PromptTemplate:
        return PromptTemplate(
            template=self.prompt_template,
            input_variables=[
                "context",
                "question",
            ],
        )

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

ARTICLE = """The Apollo program, also known as Project Apollo, was the third United States human spaceflight program carried out by the National Aeronautics and Space Administration (NASA), which accomplished landing the first humans on the Moon from 1969 to 1972.
First conceived during Dwight D. Eisenhower's administration as a three-man spacecraft to follow the one-man Project Mercury which put the first Americans in space,
Apollo was later dedicated to President John F. Kennedy's national goal of "landing a man on the Moon and returning him safely to the Earth" by the end of the 1960s, which he proposed in a May 25, 1961, address to Congress. 
Project Mercury was followed by the two-man Project Gemini (1962–66). 
The first manned flight of Apollo was in 1968.
Apollo ran from 1961 to 1972, and was supported by the two-man Gemini program which ran concurrently with it from 1962 to 1966. 
Gemini missions developed some of the space travel techniques that were necessary for the success of the Apollo missions.
Apollo used Saturn family rockets as launch vehicles. 
Apollo/Saturn vehicles were also used for an Apollo Applications Program, which consisted of Skylab, a space station that supported three manned missions in 1973–74, and the Apollo–Soyuz Test Project, a joint Earth orbit mission with the Soviet Union in 1975.
"""

prompt = PromptManager().create_prompt()
formatted_prompt = prompt.format(
    context=ARTICLE,
    question="What was the Apollo program?"
)


summary=summarizer(formatted_prompt, max_length=130, min_length=30, do_sample=False)[0]

print(summary['summary_text'])
