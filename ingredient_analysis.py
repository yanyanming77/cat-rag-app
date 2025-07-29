import streamlit as st
from PIL import Image
import pytesseract
import re
from openai import OpenAI
import os
import json
from dotenv import load_dotenv

# connect to openai API
try:
    openai.api_key = st.secrets['openai_api_key']
except:
    load_dotenv()
    openai.api_key = os.getenv('openai.api_key')

client = OpenAI(api_key = openai.api_key)

# a function to extract text from image
def extract_text_return_list(uploaded_image):
    image = Image.open(uploaded_image)
    extracted_text = pytesseract.image_to_string(image)

    extracted_text = extracted_text.strip()

    match = re.search(r'ingredients[:\s]*([\s\S]+)', extracted_text, re.IGNORECASE)
    
    if match:
        ingredients_text = match.group(1)
        
        ingredients_text = ingredients_text.replace('\n', ' ').replace('\r', ' ')
        ingredients_text = re.sub(r'\s+', ' ', ingredients_text)  # collapse multiple spaces
        
        ingredients_text = ingredients_text.rstrip(" .").strip()
        #print(f"Ingredient text:\n{ingredients_text}")
        return ingredients_text
    else:
        return ""


# a function to use LLM to analyze ingredients
import json

def analyze_by_LLM(ingredient_list):
    prompt = f"""
    You are a veterinary nutrition expert. Please analyze the nutritional quality of the following cat food based on its ingredients. Return the response in JSON format with these keys:
    - "verdict": a short, one-word quality rating (Excellent, Good, Mediocre, or Poor quality)
    - "summary": a concise summary of the overall nutritional quality of the food
    - "explanation": a dictionary with three sections by categorizing the ingredients to the following three categories, each containing a list of bullet point explanations:
        - High-Quality (e.g., named meats, whole vegetables, beneficial additives)
        - Acceptable but not ideal (e.g., vague terms, fillers)
        - Red-Flag or Harmful (e.g., artificial preservatives, unnamed meat sources)

    Please return only the JSON object.

    Ingredients:
    {ingredient_list}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful expert in cat nutrition."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        response_text = response.choices[0].message.content.strip()
        print("🧠 Raw LLM output:\n", response_text)

        # Attempt to extract JSON from the response
        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if not json_match:
            raise ValueError("No valid JSON object found in response.")

        json_str = json_match.group(0)
        response_json = json.loads(json_str)

        verdict = response_json.get("verdict", "Unknown")
        summary = response_json.get("summary", "")
        explanation_dict = response_json.get("explanation", {})

        return verdict, summary, explanation_dict

    except Exception as e:
        print(f"❌ Error during LLM analysis: {e}")
        return None, None, None


# a function to display LLM-returned results
def display_cat_food_analysis(verdict, summary, explanation_dict):
     # 1. Verdict
    st.markdown(f"""
    <div style='text-align: center; font-size: 32px; font-weight: bold; margin-top: 20px;'>
        😻 Verdict: <span style="color: green;">{verdict}</span>
        <p style='font-size: 20px;'> All verdict levels: Excellent, Good, Mideocre, Poor Quality</p>
    </div>
    """, unsafe_allow_html=True)

    # 2. Summary in light pink box
    st.markdown(f"""
    <div style='background-color: #ffe6f0; padding: 20px; border-radius: 10px; margin-top: 20px; margin-bottom: 20px;'>
        <h4>Summary of Overall Nutritional Quality</h4>
        <p>{summary}</p>
    </div>
    """, unsafe_allow_html=True)

    # 3. Explanation
    st.markdown("""
        <div style='margin-top: 30px; font-weight: bold; font-size: 24px;'>
            🔍 Explanation for Standout Ingredients
        </div>
        """, unsafe_allow_html=True)

    # high-quality ingredients
    if "High-Quality" in explanation_dict:
        st.markdown("""
            <div style='font-weight: bold; font-size: 20px; margin-top: 10px;'>
                ✅ High-Quality Ingredients
            </div>
        """, unsafe_allow_html=True)
        for line in explanation_dict["High-Quality"]:
            st.markdown(f"- {line}")

    if "Acceptable but not ideal" in explanation_dict:
        st.markdown(f"""
            <div style ='font-weight: bold; font-size: 20px; margin-top: 10px;'>
                🤔 Acceptable but Not Ideal 
            </div>""", unsafe_allow_html=True)
        for line in explanation_dict["Acceptable but not ideal"]:
            st.markdown(f"- {line}")

    if "Red-Flag or Harmful" in explanation_dict:
        st.markdown("""
            <div style='font-weight: bold; font-size: 20px; margin-top: 10px;'> 
                🚫 Red-Flag Ingredients" 
            </div>""", unsafe_allow_html=True)
        for line in explanation_dict["Red-Flag or Harmful"]:
            st.markdown(f"- {line}")

def run_analyze_ingredients():
    st.markdown("""
        <div style='text-align: center;'>
            <h1>😺 Analyze Ingredients of Cat Food</h1>
        </div>
        """, unsafe_allow_html=True)

    # add feature to allow user's upload of plant or nutrition fact image file
    uploaded_image = st.file_uploader("📷 Upload Image to analyze the ingredients of the cat food", type=["png", "jpg", "jpeg"], key="img_upload")
    if uploaded_image:
        # display the smaller image
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.image(uploaded_image, caption="Uploaded Image", width=300)
        st.markdown("</div>", unsafe_allow_html=True)

        # add a spinner while LLM is thinking
        with st.spinner("🔍 Extracting ingredients and analyzing with LLM..."):
            # extract ingredient list
            ingredient_list = extract_text_return_list(uploaded_image)
            # perform analysis using LLM
            if not ingredient_list:
                st.warning("⚠️ No ingredients detected, please try another image.")
            else:
                verdict, summary, explanation_dict = analyze_by_LLM(ingredient_list)


        st.subheader("📋 Ingredient Analysis Summary")
        st.markdown(
                f"<div style='margin-bottom: 1px; '>{display_cat_food_analysis(verdict, summary, explanation_dict)}</div>",
                unsafe_allow_html=True
            )
            

