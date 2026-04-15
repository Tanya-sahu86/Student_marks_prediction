import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


student= {
  "User_ID": ["U001","U002","U003","U004","U005","U006","U007","U008","U009","U010","U011","U012","U013","U014","U015"],
  "Age": [33,40,25,30,28,35,45,23,31,29,38,27,34,26,32],
  "Gender":["Male","Female","Other","Female","Male","Other","Other","Male","Female","Female","Male","Other","Male","Male","Female"],
  "Daily_Screen_Time":[5,6,3,7,8,4,9,5,6,4,2,7,8,4,7],
  "Sleep_Quality":[6,7,8,5,7,4,8,8,5,6,4,9,5,6,7],
  "Stress_Level":[6,4,7,5,3,9,2,6,3,6,8,4,6,5,7],
  "Days_Without_Social_Media":[1,2,3,2,3,1,2,0,1,0,0,2,1,3,2],
  "Exercise_Frequently":[4,2,6,3,5,4,4,4,3,6,7,2,5,4,3],
  "Social_Media_Platform":["Facebook","Instragram","Twitter","Instragram","Twitter","Linkdln","Tiktok","Facebook","Tiktok","Linkdln","Instragram","Facebook","Twitter","Tiktok","Linkdln"],
  "Happiness_Index":[7,4,5,3,7,6,2,6,1,5,9,4,6,5,6]
}
df= pd.DataFrame(student)
print(df)

