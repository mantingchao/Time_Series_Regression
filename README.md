# Time_Series_Regression
- 使用新竹地區2019年10~12月之空氣品質資料，進行時間序列分析&迴歸預測pm2.5值
- 使用10和11月資料當作訓練集，12月之資料當作測試集
- 將前六小時的汙染物數據做為特徵，未來第一個小時/未來第六個小時的pm2.5數據為預測目標，使用兩種模型 Linear Regression 和 Random Forest Regression 建模並計算MAE
