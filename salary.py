def check_salary_level(yrs_of_xp):
    if yrs_of_xp >= 9:
        return 200000
    elif yrs_of_xp < 9 and yrs_of_xp >= 7:
        return 160000
    elif yrs_of_xp < 7 and yrs_of_xp >= 5:
        return 120000
    elif yrs_of_xp < 5 and yrs_of_xp >= 3:
        return 80000
    elif yrs_of_xp < 3 and yrs_of_xp >= 1:
        return 40000
    else:
        return "Upskill ka muna!"


my_expi = 3.5
result = check_salary_level(my_expi)
print(result)