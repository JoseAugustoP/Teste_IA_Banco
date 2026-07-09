import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.3546/dados?formato=json&dataInicial=07/07/2016&dataFinal=07/07/2026"
resposta = requests.get(url)
dados = resposta.json()

df = pd.DataFrame(dados)
df["data"] = pd.to_datetime(df["data"], dayfirst=True)
df["valor"] = df["valor"].astype(float)
df["dias"] = (df["data"] - df["data"].min()).dt.days

x = df[["dias"]]
y = df["valor"]

x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.2,random_state=42
)

modelo = LinearRegression()
modelo.fit(x_train, y_train)
previsoes = modelo.predict(x_test)
previsao_completa = modelo.predict(df[["dias"]]) 

plt.plot(df["data"], df["valor"], color="gray", alpha=0.5, label="Dados reais")
plt.plot(df["data"], previsao_completa, color="blue", linewidth=2, label="Previsão da IA")
plt.xlabel("Data")
plt.ylabel("Valor (R$)")
plt.title("Reservas Internacionais do Brasil")
plt.legend()
plt.show()


