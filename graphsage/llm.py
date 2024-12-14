from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def load_local_llm():
    model_name = "microsoft/Phi-3-mini-4k-instruct" 
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto"
    )
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer#,
        #device=0  # Use GPU (if available) set device id; change to -1 for CPU
    )
    return HuggingFacePipeline(pipeline=llm_pipeline)

def generate_recommendations(preferences, top_recipes):
    """
    Generate personalized recipe descriptions using LangChain and a local LLM.

    :param preferences: User preferences as a string.
    :param top_recipes: List of top recipe names.
    :return: LLM-generated recommendation descriptions.
    """
    llm = load_local_llm()
    template = """
    Using the LLM, based on the user's preferences: {preferences}, here are some top recommended recipes:
    {recipes}

    Provide personalized descriptions for these recipes.
    """
    prompt = PromptTemplate(input_variables=["preferences", "recipes"], template=template)
    chain = LLMChain(llm=llm, prompt=prompt)
    recipes_str = "\n".join(top_recipes)
    response = chain.run(preferences=preferences, recipes=recipes_str)
    return response.strip()
