# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 08:57:37 2019

@author: pverschuuren
"""
    
# Compare two dates.
def same_date(date1, date2):
    
    if date1 == '':
        return 0.0
    if date2 == '':
        return 0.0
    if date1 == 'nan':
        return 0.0
    if date2 == 'nan':
        return 0.0
    
    # Convert the date inputs to strings.
    date1 = str(date1)
    date2 = str(date2)
    
    # Strip the date strings from any format specific characters.
    date1 = date1.replace("/","")
    date2 = date2.replace("/","")
    date1 = date1.replace("-","")
    date2 = date2.replace("-","")
    
    if date1 == date2:
        return 1.0
    else:
        return 0.0

# Check if two numbers are the same.
def same_num(num1, num2):
    
    if isinstance(num1, str):
        num1 = num1.replace('"','')
    if isinstance(num2, str):
        num2 = num2.replace('"','')
        
    
    num1 = float(num1)
    num2 = float(num2)
    
    if num1 == num2:
        return 1.0
    else:
        return 0.0
