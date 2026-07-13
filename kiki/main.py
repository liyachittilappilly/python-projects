from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet

from tools import search_tool

load_dotenv()


# ---------- Output Structure ----------

class ResearchResponse(BaseModel):
    title: str
    introduction: str
    historical_background: str
    important_facts: list[str]
    significance: str
    conclusion: str
    sources: list[str]


parser = PydanticOutputParser(
    pydantic_object=ResearchResponse
)


# ---------- Gemini ----------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)


# ---------- User Query ----------

query = input("What can I help you research?\n> ")

print("\nSearching web...")
web_info = search_tool.run(query)


# ---------- Prompt ----------

prompt = ChatPromptTemplate.from_template(
"""
You are an expert research assistant.

Write a detailed report.

Include:

1. Introduction
2. Historical Background
3. Important Facts (5-10 bullet points)
4. Significance
5. Conclusion
6. Sources

Use the following web information:

{web_info}

Question:

{query}

Return ONLY in this format:

{format_instructions}
"""
)

prompt = prompt.partial(
    format_instructions=parser.get_format_instructions()
)

chain = prompt | llm | parser


# ---------- Generate Response ----------

response = chain.invoke(
    {
        "query": query,
        "web_info": web_info
    }
)


# ---------- PDF ----------

doc = SimpleDocTemplate("research_report.pdf")

styles = getSampleStyleSheet()

story = []


# Title

story.append(
    Paragraph(
        response.title,
        styles['Title']
    )
)

story.append(Spacer(1, 20))


# Introduction

story.append(
    Paragraph(
        "Introduction",
        styles['Heading1']
    )
)

story.append(
    Paragraph(
        response.introduction,
        styles['BodyText']
    )
)

story.append(Spacer(1, 20))


# Historical Background

story.append(
    Paragraph(
        "Historical Background",
        styles['Heading1']
    )
)

story.append(
    Paragraph(
        response.historical_background,
        styles['BodyText']
    )
)

story.append(Spacer(1, 20))


# Important Facts

story.append(
    Paragraph(
        "Important Facts",
        styles['Heading1']
    )
)

for fact in response.important_facts:
    story.append(
        Paragraph(
            "• " + fact,
            styles['BodyText']
        )
    )

story.append(Spacer(1, 20))


# Significance

story.append(
    Paragraph(
        "Significance",
        styles['Heading1']
    )
)

story.append(
    Paragraph(
        response.significance,
        styles['BodyText']
    )
)

story.append(Spacer(1, 20))


# Conclusion

story.append(
    Paragraph(
        "Conclusion",
        styles['Heading1']
    )
)

story.append(
    Paragraph(
        response.conclusion,
        styles['BodyText']
    )
)

story.append(Spacer(1, 20))


# Sources

story.append(
    Paragraph(
        "Sources",
        styles['Heading1']
    )
)

for source in response.sources:
    story.append(
        Paragraph(
            "• " + source,
            styles['BodyText']
        )
    )


doc.build(story)

print("\nPDF saved as research_report.pdf")