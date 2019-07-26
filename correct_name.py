import numpy as np
import editdistance
import re




patterns = {
    '[àáảãạăắằẵặẳâầấậẫẩ]': 'a',
    '[đ]': 'd',
    '[èéẻẽẹêềếểễệ]': 'e',
    '[ìíỉĩị]': 'i',
    '[òóỏõọôồốổỗộơờớởỡợ]': 'o',
    '[ùúủũụưừứửữự]': 'u',
    '[ỳýỷỹỵ]': 'y'
}



def notonizer(text):
    """
    function to convert Vietnamese toned text to none tone
    :param text: input text
    :return: none tone text
    """
    output = text
    for regex, replace in patterns.items():
        output = re.sub(regex, replace, output)
        output = re.sub(regex.upper(), replace.upper(), output)
    return output



converted_family_names =['An', 'Ao', 'Biện', 'Bàn', 'Bàng', 'Bành', 'Bá', 'Bì', 'Bình',
       'Bùi', 'Bạc', 'Bạch', 'Bảo', 'Bế', 'Bồ', 'Ca', 'Cam', 'Cao',
       'Chiêm', 'Chu/Châu', 'Chung', 'Chúng', 'Chương', 'Chế', 'Chử',
       'Cung', 'Cái', 'Cáp', 'Cát', 'Công', 'Cù', 'Cảnh', 'Cấn', 'Cầm',
       'Cống', 'Cồ', 'Cổ', 'Cự', 'Danh', 'Diêm', 'Diệp', 'Doãn', 'Dã',
       'Dư', 'Dương', 'Giang', 'Giao', 'Giàng', 'Giáp', 'Giả', 'Hi',
       'Hoa', 'Hoàng', 'Huỳnh', 'Hà', 'Hàn', 'Hán', 'Hình', 'Hùng',
       'Hướng', 'Hạ', 'Hầu', 'Hề', 'Hồ', 'Hồng', 'Hứa', 'Kha', 'Khiếu',
       'Khoa', 'Khu', 'Khuất', 'Khâu', 'Khúc', 'Khương', 'Khổng', 'Kim',
       'Kiều', 'Kiểu', 'La', 'Liêu', 'Liễu', 'Luyện', 'Lâm', 'Lã', 'Lãnh',
       'Lê', 'Lò', 'Lý', 'Lăng', 'Lư', 'Lưu', 'Lương', 'Lạc', 'Lại',
       'Lều', 'Lỗ', 'Lộ', 'Lục', 'Ma', 'Mai', 'Mang', 'Mâu', 'Mã', 'Mạc',
       'Mạch', 'Mạnh', 'Mẫn', 'Mộc', 'Mục', 'Nghiêm', 'Nghị', 'Nguyễn',
       'Ngân', 'Ngô', 'Ngọ', 'Ngọc', 'Ngụy', 'Nhan', 'Nhâm', 'Nhữ',
       'Ninh', 'Nông', 'Ong', 'Phan', 'Phi', 'Phí', 'Phó', 'Phùng', 'Phú',
       'Phương', 'Phạm', 'Quàng', 'Quách', 'Quản', 'Sùng', 'Sơn', 'Sầm',
       'Sử', 'Thang', 'Thi', 'Thiều', 'Thoa', 'Thành', 'Thào', 'Thái',
       'Thân', 'Thôi', 'Thạch', 'Thẩm', 'Thập', 'Thịnh', 'Thục', 'Ti',
       'Tinh', 'Tiêu', 'Tiếp', 'Trang', 'Tri', 'Triệu', 'Trà', 'Trác',
       'Trình', 'Trưng', 'Trương', 'Trần', 'Trịnh', 'Tào', 'Tán', 'Tòng',
       'Tô', 'Tôn', 'Tông', 'Tăng', 'Tạ', 'Tống', 'Từ', 'Ung', 'Uông',
       'Vi', 'Viêm', 'Viên', 'Võ', 'Văn', 'Vũ', 'Vưu', 'Vương', 'Vạn',
       'Xa', 'Xung', 'Yên', 'Ánh', 'Ân', 'Âu Dương', 'Ông', 'Đan', 'Đinh',
       'Điêu', 'Điền', 'Đoàn', 'Đàm', 'Đào', 'Đèo', 'Đôn', 'Đương',
       'Đường', 'Đậu', 'Đặng', 'Đống', 'Đồ', 'Đồng', 'Đổng', 'Đỗ', 'Đới',
       'Đức', 'Ưng', 'Ấu', 'Ứng']


converted_first_names =['An', 'Anh', 'Bào', 'Bình', 'Bích', 'Băng', 'Bạch', 'Bảo', 'Bắc',
       'Bằng', 'Bổng', 'Bửu', 'Ca', 'Canh', 'Cao', 'Chi', 'Chinh',
       'Chiêu', 'Chiến', 'Chiểu', 'Chung', 'Chuyên', 'Châu', 'Chính',
       'Chương', 'Chưởng', 'Chấn', 'Cung', 'Cát', 'Công', 'Cúc', 'Cơ',
       'Cương', 'Cường', 'Cảnh', 'Cầm', 'Cần', 'Cẩn', 'Danh', 'Dao', 'Di',
       'Dinh', 'Diễm', 'Diệp', 'Diệu', 'Doanh', 'Du', 'Dung', 'Duy',
       'Duyên', 'Duyệt', 'Duệ', 'Dân', 'Dũng', 'Dương', 'Dạ', 'Dụng',
       'Gia', 'Giang', 'Giao', 'Giác', 'Giáp', 'Hải', 'Hiên', 'Hiếu', 'Hiền',
       'Hiển', 'Hiệp', 'Hoa', 'Hoan', 'Hoài', 'Hoàn', 'Hoàng', 'Hoán',
       'Hoạt', 'Huy', 'Huynh', 'Huyền', 'Huấn', 'Huệ', 'Huỳnh', 'Hà',
       'Hàm', 'Hành', 'Hào', 'Hân', 'Hãn', 'Hòa', 'Hùng', 'Hưng', 'Hương',
       'Hường', 'Hưởng', 'Hạ', 'Hạnh', 'Hảo', 'Hậu', 'Hằng', 'Học',
       'Hồng', 'Hội', 'Hợp', 'Hữu', 'Hỷ', 'Kha', 'Khai', 'Khang', 'Khanh',
       'Khiêm', 'Khiếu', 'Khoa', 'Khoan', 'Khoát', 'Khuyên', 'Khuê',
       'Khánh', 'Khê', 'Khôi', 'Khương', 'Khải', 'Kim', 'Kiên', 'Kiếm',
       'Kiều', 'Kiện', 'Kiệt', 'Kê', 'Kính', 'Kỳ', 'Kỷ', 'Lai', 'Lam',
       'Lan', 'Linh', 'Liêm', 'Liên', 'Liễu', 'Liệt', 'Loan', 'Long',
       'Luân', 'Luận', 'Luật', 'Ly', 'Lâm', 'Lân', 'Lý', 'Lăng', 'Lĩnh',
       'Lương', 'Lạc', 'Lập', 'Lễ', 'Lệ', 'Lộ', 'Lộc', 'Lợi', 'Lực',
       'Mai', 'Mi', 'Minh', 'Miên', 'My', 'Mạnh', 'Mẫn', 'Mỹ', 'Nam',
       'Nga', 'Nghi', 'Nghiêm', 'Nghiệp', 'Nghĩa', 'Nghị', 'Nguyên',
       'Nguyệt', 'Ngà', 'Ngân', 'Ngôn', 'Ngạn', 'Ngọc', 'Nhi', 'Nhiên',
       'Nhiệm', 'Nhu', 'Nhung', 'Nhuận', 'Nhàn', 'Nhạn', 'Nhân', 'Nhã', 'Nhơn',
       'Như', 'Nhượng', 'Nhạn', 'Nhất', 'Nhật', 'Ninh', 'Năng', 'Nương',
       'Nữ', 'Oanh', 'Phi', 'Phong', 'Phu', 'Pháp', 'Phát', 'Phú', 'Phúc',
       'Phương', 'Phước', 'Phượng', 'Phụng', 'Quang', 'Quyên', 'Quyết',
       'Quyền', 'Quân', 'Quý', 'Quảng', 'Quế', 'Quốc', 'Quỳnh', 'Sa',
       'San', 'Sang', 'Sinh', 'Siêu', 'Sáng', 'Sâm', 'Sĩ', 'Sơn', 'Sương',
       'Sử', 'Sỹ', 'Thanh', 'Thi', 'Thiên', 'Thiện', 'Thoa', 'Thoại',
       'Thu', 'Thuyết', 'Thuần', 'Thuận', 'Thuật', 'Thy', 'Thành', 'Thái',
       'Thêu', 'Thông', 'Thùy', 'Thúc', 'Thúy', 'Thơ', 'Thư', 'Thương',
       'Thường', 'Thạc', 'Thạch', 'Thảo', 'Thắm', 'Thắng', 'Thế', 'Thể',
       'Thịnh', 'Thọ', 'Thống', 'Thời', 'Thục', 'Thụy', 'Thủy', 'Thực',
       'Tiên', 'Tiến', 'Tiếp', 'Tiền', 'Tiển', 'Toàn', 'Toại', 'Toản',
       'Trang', 'Tranh', 'Trinh', 'Triết', 'Triều', 'Triệu', 'Trung',
       'Trà', 'Trác', 'Tráng', 'Trâm', 'Trân', 'Trình', 'Trí', 'Trúc',
       'Trương', 'Trường', 'Trưởng', 'Trạch', 'Trầm', 'Trọng', 'Trụ',
       'Trực', 'Tuyến', 'Tuyết', 'Tuyền', 'Tuyển', 'Tuấn', 'Tuệ', 'Ty',
       'Tài', 'Tâm', 'Tân', 'Tín', 'Tính', 'Tùng', 'Tú', 'Tường', 'Tấn',
       'Tịnh', 'Tổ', 'Tụ', 'Từ', 'Uy', 'Uyên', 'Uyển', 'Vi', 'Vinh',
       'Viên', 'Việt', 'Vu', 'Vy', 'Vân', 'Võ', 'Văn', 'Vĩ', 'Vĩnh', 'Vũ',
       'Vương', 'Vượng', 'Vịnh', 'Vọng', 'Vỹ', 'Xuyến', 'Xuân', 'Yên',
       'Yến', 'xanh', 'Ái', 'Án', 'Ánh', 'Ân', 'Ðan', 'Ðiền', 'Ðiệp',
       'Ðoan', 'Ðoàn', 'Ðài', 'Ðàn', 'Ðào', 'Ðình', 'Ðôn', 'Ðông', 'Ðăng',
       'Ðường', 'Ðại', 'Ðạo', 'Ðạt', 'Ðệ', 'Ðịnh', 'Ðồng', 'Ðộ', 'Ðức',
       'Ý', 'Đan', 'Đào', 'Đăng', 'Đức', 'Ẩn']

converted_middle_names=['An', 'Anh', 'Ban', 'Bá', 'Bách', 'Bình', 'Bích', 'Băng', 'Bạch',
       'Bảo', 'Bằng', 'Bội', 'Bửu', 'Bữu', 'Cam', 'Cao', 'Chi', 'Chiêu',
       'Chiến', 'Chung', 'Chuẩn', 'Chánh', 'Chí', 'Chính', 'Chấn', 'Chế',
       'Cát', 'Công', 'Cương', 'Cường', 'Cảnh', 'Cẩm', 'Danh', 'Di',
       'Diên', 'Diễm', 'Diệp', 'Diệu', 'Duy', 'Duyên', 'Dân', 'Dã',
       'Dũng', 'Dương', 'Dạ', 'Gia', 'Giang', 'Giao', 'Giáng', 'Hiếu',
       'Hiền', 'Hiểu', 'Hiệp', 'Hoa', 'Hoài', 'Hoàn', 'Hoàng', 'Hoạ',
       'Huy', 'Huyền', 'Huân', 'Huệ', 'Huỳnh', 'Hà', 'Hàm', 'Hào', 'Hải', 'Hán',
       'Hòa', 'Hùng', 'Hưng', 'Hương', 'Hướng', 'Hạ', 'Hạc', 'Hạnh',
       'Hạo', 'Hải', 'Hảo', 'Hằng', 'Họa', 'Hồ', 'Hồng', 'Hữu', 'Khai',
       'Khang', 'Khiết', 'Khoa', 'Khuyến', 'Khuê', 'Khánh', 'Khôi',
       'Khúc', 'Khương', 'Khả', 'Khải', 'Khắc', 'Khởi', 'Kim', 'Kiên',
       'Kiến', 'Kiết', 'Kiều', 'Kiệt', 'Kỳ', 'Lam', 'Lan', 'Linh', 'Liên',
       'Liễu', 'Loan', 'Long', 'Ly', 'Lâm', 'Lê', 'Lưu', 'Lương', 'Lạc',
       'Lập', 'Lệ', 'Lộc', 'Lục', 'Mai', 'Minh', 'Mạnh', 'Mậu', 'Mộc',
       'Mộng', 'Mỹ', 'Nam', 'Nghi', 'Nghĩa', 'Nghị', 'Nguyên', 'Nguyết',
       'Nguyễn', 'Nguyệt', 'Ngân', 'Ngọc', 'Nhan', 'Nhân', 'Nhã', 'Như',
       'Nhất', 'Nhật', 'Niệm', 'Oanh', 'Phi', 'Phong', 'Phú', 'Phúc',
       'Phương', 'Phước', 'Phượng', 'Phục', 'Phụng', 'Quang', 'Quyết',
       'Quân', 'Quý', 'Quảng', 'Quế', 'Quốc', 'Quỳnh', 'Sao', 'Song',
       'Sông', 'Sĩ', 'Sơn', 'Sương', 'Sỹ', 'Thanh', 'Thi', 'Thiên',
       'Thiếu', 'Thiều', 'Thiện', 'Thiệu', 'Thu', 'Thuần', 'Thuận', 'Thy',
       'Thành', 'Thái', 'Thông', 'Thùy', 'Thúy', 'Thăng', 'Thơ', 'Thư',
       'Thương', 'Thường', 'Thượng', 'Thạch', 'Thảo', 'Thất', 'Thắng',
       'Thế', 'Thịnh','THỊ', 'Thống', 'Thời', 'Thụ', 'Thục', 'Thụy', 'Thủy',
       'Tinh', 'Tiên', 'Tiến', 'Tiền', 'Tiểu', 'Toàn', 'Trang', 'Triều',
       'Triển', 'Triệu', 'Trung', 'Trà', 'Trâm', 'Trân', 'Trí', 'Trúc',
       'Trường', 'Trầm', 'Trọng', 'Tuyết', 'Tuyền', 'Tuấn', 'Tuệ', 'Tài',
       'Tâm', 'Tân', 'Tích', 'Tôn', 'Tùng', 'Tùy', 'Tú', 'Túy', 'Tường',
       'Tạ', 'Tấn', 'Tất', 'Tịnh', 'Tố', 'Từ', 'Uy', 'Uyên', 'Uyển', 'Vi',
       'Vinh', 'Viết', 'Viễn', 'Việt', 'Vy', 'Vàng', 'Vành', 'Vân', 'Văn',
       'Vĩnh', 'Vũ', 'Vương', 'Vạn', 'Xuyến', 'Xuân', 'Yên', 'Yến', 'Ái',
       'Ánh', 'Ân', 'Ðan', 'Ðinh', 'Ðoan', 'Ðoàn', 'Ðài', 'Ðình', 'Ðông',
       'Ðăng', 'đặng', 'Ðại', 'Ðạt', 'Ðắc', 'Ðịnh', 'Ðồng', 'Ðức', 'Ý', 'Đan',
       'Đinh', 'Đoan', 'Đài', 'Đông', 'Đăng', 'Đơn', 'Đức', 'Ấu', 'An']


famous_family_names=['nguyễn','trần','lê','phạm','hoàng','huỳnh','phan',
                     'vũ','võ','đặng','bùi','đỗ','hồ','ngô','dương','lý']



converted_family_names =[item.lower() for item in converted_family_names]
for index in range(len(converted_family_names)):
  if converted_family_names[index][0]=='ð':converted_family_names[index]='đ'+converted_family_names[index][1:]

converted_middle_names=[item.lower() for item in converted_middle_names]
converted_first_names=[item.lower() for item in converted_first_names]

def correct_name(check_name_input, mode='family_name'):
  assert mode in ['family_name','middle_name','first_name'], 'wrong mode'
  check_name=check_name_input.lower()
  if mode == 'family_name':
    editdistances= list(map(lambda name : editdistance.eval(name,check_name ),converted_family_names))
    map_dictionaries = converted_family_names
  elif mode=='middle_name':
    editdistances= list(map(lambda name : editdistance.eval(name,check_name ),converted_middle_names))
    map_dictionaries = converted_middle_names
  else:
    editdistances = list(map(lambda name: editdistance.eval(name, check_name), converted_first_names))
    map_dictionaries = converted_first_names

  editdistances_np = np.array(editdistances)
  idx=editdistances_np.argsort()
  min_np = np.min(editdistances_np)
  results = []
  best_result=map_dictionaries[idx[0]]
  if editdistances_np[idx[0]]==0:
      return [best_result]

  for index in idx:
    if(editdistances[index]==min_np or editdistances[index]==min_np+1):
      results.append(map_dictionaries[index])

  results_no_tone=[notonizer(item) for item in results]
  check_name_no_tone=notonizer(check_name)
  res=[]

  for index in range(len(results_no_tone)):
    if results_no_tone[index]==check_name_no_tone:
      res.append(results[index])

  if res: return res
  return results[0:3]

