# RW_BBQ_Script
Python script to make an LLM handle BBQ benchmark questions with different personas.

Some important notes about our setup in LM studio:
  1, The script is designed to handle 1 model at a time, this was convenient for us computationally wise. This means you need to change the model name manually in the python script before you run it.
  2, Since we used Qwen models, which are reasoning models, we had to add this line in the "Inference" option in LMstudio:
  {%- set enable_thinking = false %} 
  This disabled the written-out reasoning process of the models and they were able to answer a single A/B/C.
  
  
