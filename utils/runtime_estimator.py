vsp_combination = 11*7*9*4*4*5
mission2_combination = 1
mission3_combination = 1
vsp_server_number = 256
mission_server_number = 256

total_time = (vsp_combination * 15 / vsp_server_number) + (vsp_combination/mission_server_number)*(mission2_combination*1.8 + mission3_combination*0)
total_time = total_time/3600 

print(f"\ntotal running time = {total_time} hour")