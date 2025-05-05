vsp_combination = 41*16*11*4
mission2_combination = 1
mission3_combination = 1
vsp_server_number = 32
mission_server_number = 1

total_time = (vsp_combination * 15 / vsp_server_number) + (vsp_combination/mission_server_number)*(mission2_combination*0.18 + mission3_combination*1)
total_time = total_time/3600 

print(f"\ntotal running time = {total_time} hour")