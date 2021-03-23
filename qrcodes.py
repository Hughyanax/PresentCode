#原始二维码编码
import qrcode
data = '生日快乐啊朋友！\n希望你上分顺利\n玩游戏就要开开心心'
#带图片的编码方式
qr = qrcode.QRCode(
    version=5,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=8,
    border=4)
qr.add_data(data)
qr.make(fit=True)
img = qr.make_image()
# 保存二维码
img.save("my_qrcode.gif")
