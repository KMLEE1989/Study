a_dict = {"023943" : {"종목명": "섬광글라스", '등락율(%)': 0.0, "고가대비(%)":-0.97,'보유수량': 0,
          '현재가': 41050, '(최우선)매도호가': 41050}}

# print(a_dict.keys())
# print(a_dict['023943'].keys())
# dict_keys(['023943'])
# dict_keys(['종목명', '등락율(%)', '고가대비(%)', '보유수량', '현재가', '(최우선)매도호가'])

# print(a_dict.get('023943').get('고가대비(%)'))
# -0.97

# 섬광글라스


# for key, value in a_dict.items():
#     print("키: %s, 값: %s" % (key, value))

# 키: 종목명, 값: 섬광글라스
# 키: 등락율(%), 값: 0.0
# 키: 고가대비(%), 값: -0.97
# 키: 보유수량, 값: 0
# 키: 현재가, 값: 41050
# 키: (최우선)매도호가, 값: 41050

# a_dict.update({'종목명':'키움증권'})
# a_dict.update({'보유수량':5, '현재가':5000})
# a_dict.update({'없는키': 1491591})
# print(a_dict)
# {'종목명': '키움증권', '등락율(%)': 0.0, '고가대비(%)': -0.97, '보유수량': 5, '현재가': 5000, '(최우선)매도호가': 41050}
# {'종목명': '키움증권', '등락율(%)': 0.0, '고가대비(%)': -0.97, '보유수량': 5, '현재가': 5000, '(최우선)매도호가': 41050, '없는키': 1491591}
#
# a_dict['종목명'] = '키움증권'
# a_dict['없는키'] = 151512

# print(a_dict)
# {'종목명': '키움증권', '등락율(%)': 0.0, '고가대비(%)': -0.97, '보유수량': 0, '현재가': 41050, '(최우선)매도호가': 41050}
# {'종목명': '키움증권', '등락율(%)': 0.0, '고가대비(%)': -0.97, '보유수량': 0, '현재가': 41050, '(최우선)매도호가': 41050, '없는키': 151512}
