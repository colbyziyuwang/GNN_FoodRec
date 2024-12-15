from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def load_local_llm():
    """
    Load Google's Flan-T5 model for recommendation generation.
    """
    model_name = "google/flan-t5-large"  # You can change this to other Flan-T5 sizes like flan-t5-base, flan-t5-xl
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        device_map="auto",  # Automatically allocate the model on the available device (GPU/CPU)
        torch_dtype="auto"  # Optimize tensor operations based on available hardware
    )
    llm_pipeline = pipeline(
        "text2text-generation",  # Use text2text-generation pipeline for Flan-T5
        model=model,
        tokenizer=tokenizer
    )
    return HuggingFacePipeline(pipeline=llm_pipeline)


def generate_recommendations(preferences, top_recipes):
    """
    Use LLM to generate the best recommendation and a justification from the top 10 recipes.

    :param preferences: User preferences as a string.
    :param top_recipes: List of top 10 recipe names from similarity scoring.
    :return: LLM-generated single recommendation with justification.
    """
    llm = load_local_llm()

    # Updated structured prompt
    template = """
    From the following preferences: {preferences} and recipes:
    {recipes}
    Please choose the best recipe for the preferences - only provide the recipe name and reason
    """
    prompt = PromptTemplate(input_variables=["preferences", "recipes"], template=template)
    chain = LLMChain(llm=llm, prompt=prompt)

    # Convert the recipe list into a numbered string
    #recipes_str = "\n".join([f"{i + 1}. {recipe}" for i, recipe in enumerate(top_recipes)])
    
    # Run the chain with the structured prompt
    response = chain.run(preferences=preferences, recipes=top_recipes)

    return response.strip()



