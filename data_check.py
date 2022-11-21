import pandas as pd

def Chatbot_Data(num_data=None):
    Chatbot_Data = pd.read_csv("ChatBotData.csv")
    if num_data is not None :
        Chatbot_Data = Chatbot_Data[:num_data] # 테스트용으로 쓸 경우 
        return Chatbot_Data
    
    else :
        Chatbot_Data = Chatbot_Data
        return Chatbot_Data

