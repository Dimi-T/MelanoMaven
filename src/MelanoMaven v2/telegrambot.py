from telegram.ext import Application, CommandHandler, MessageHandler, filters
from predict import get_predict
from zoomcroptransform import prepare
import glob
import os
import shutil
import time

class Timer:
    def __init__(self):
        self.ids = {}

    def add_id(self, id):
        self.ids[id] = time.time()        

    def get_time(self, id):
        return time.time() - self.ids[id]
    
    def exists(self, id):
        return id in self.ids

    def reset_time(self, id):
        self.ids[id] = time.time()

class EnableProcess:
    def __init__(self):
        self.enable_process = False

    def set_enable_process(self, value):
        self.enable_process = value

    def get_enable_process(self):
        return self.enable_process


first_time = {}

enable = EnableProcess()
timer = Timer()

Token = '6144801054:AAGlpgqLF8EC_yj5wNSEpMaqjRLmhMJn7UE'
Username = '@Melano_Maven_v2_bot'
img_types = ['heic', 'heics', 'HEIC', 'HEICS', 'png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG']


async def start(update, context):

    enable.set_enable_process(False)

    await update.message.reply_text(
    """
    Hello! Welcome to MelanoMaven v2.0.0!
I am a telegram bot that can give you a prediction of whether or not you have melanoma based on a picture of your lesion!
    """
    )

    with open('../../resources/users.log', 'r') as f:
        if str(update.message.chat_id) in f.read():
            first_time[str(update.message.chat_id)] = False
        else:
            with open('../../resources/users.log', 'a') as f:
                f.write(str(update.message.chat_id) + '\n')
            timer.add_id(str(update.message.chat_id))
            first_time[str(update.message.chat_id)] = True



async def help(update, context):

    await update.message.reply_text(
    """
    /start - Welcoming message and short description of the bot
/help -  List of commands and their functions
/info - Info about MelanoMaven v2.0.0
/predict - Get a prediction
    """
    )

async def info(update, context):

    await update.message.reply_text(
    """
        MelanoMaven v2.0.0
    Created by: @DimitrieTanasescu



    MelanoMaven is a research project that aims to create an automatic tool that can predict whether or not a person has melanoma based on a picture of their lesion.
    The Machine Learning model implemented is a Convolutional Neural Network (CNN) based on the VGG16 architecture and it is trained on the ISIC 2017 and ISIC 2020 datasets.

    MelanoMaven is not a substitute for a proper medical examination and should not be used as such. The project is still in research and development phase and should not be used for medical purposes.
    If you suspect you may have any form of skin cancer, please consult a dermatologist and an oncologist immediately.
    """
    )

async def predict(update, context):
    
    await update.message.reply_text("Certainly! Send me a picture of your lesion and I will tell you if you have a melanoma or not!\nPlease take the picture in good lighting conditions and make sure the lesion is in focus.\nWith a default zoom factor of x1.0, the photo should be taken from a distance of about 25cm (10 inches) from the lesion.\nThe photo should look something like this:")
    await update.message.reply_photo(open('../../resources/example.jpg', 'rb'))

    if first_time[str(update.message.chat_id)]:
        timer.reset_time(str(update.message.chat_id))
        first_time[str(update.message.chat_id)] = False
        wait = -1
        #print(first_time[str(update.message.chat_id)])
    else:
        wait = 20

    if timer.get_time(str(update.message.chat_id)) > wait:
        enable.set_enable_process(True)
    else:
        enable.set_enable_process(False)
        await update.message.reply_text("Please wait a few seconds before sending another predict command.")

async def donwload_image(update, context):
    
        image = await update.message.effective_attachment[-1].get_file()
        await image.download_to_drive()

        new_path = None
    
        for img_type in img_types:
            m_file = glob.glob("*.{}".format(img_type))
            if m_file == []:
                continue        
            else:
                new_path = "../../input/image.{}".format(img_type)
                shutil.move(m_file[0], new_path)

        if new_path == None:
            await update.message.reply_text("Please send a valid image file.")

        return new_path

async def get_report(update, context):

    with open('../../resources/med_users.log', 'r') as f:
        if str(update.message.chat_id) not in f.read():
            await update.message.reply_text("You are not authorized to use this command.")
            return
        else:
            await update.message.reply_text("Please wait while I generate the report.")
            report = ""
            with open('../../resources/results.log', 'r') as f:
                for line in f.readlines():
                    report += line
            await update.message.reply_text(report)

async def handle_image(update, context):
    
    if not enable.get_enable_process():
        await update.message.reply_text("If you need a prediction, please use the /predict command.")
        return

    filename = await donwload_image(update, context)

    if filename == None:
        return

    filename = prepare(filename)
    #print('Image prepared')
    pred = get_predict(filename)
    #print('Prediction made')

    if pred == "Benign":
        result = "According to my estimation, your lesion is benign (non-cancerous).\nPlease keep in mind that if you have any suspicions about your lesion, you should consult a dermatologist."

    elif pred == "Malignant":
        result = "According to my estimation, your lesion might indicate a the presence of melanoma (cancerous tissue).\nI recommend you to consult a dermatologist or an oncologist for a thorough examination."
    os.remove(filename)
    await update.message.reply_text(result)

    with open('../../resources/results.log', 'a') as f:
        f.write(str(update.message.chat.first_name) + ' ' + str(update.message.chat.last_name) + ': ' + pred + '\n')

    enable.set_enable_process(False)

async def error(update, context):
    print(f'Update {update} caused error {context.error}')

async def handle_text(update, context):
    await update.message.reply_text("Sorry, I don't understand. Please use the /help command for a list of commands.")
    
if __name__ == '__main__':

    try:
        os.remove('../../resources/users.log')
    except:
        pass

    open('../../resources/users.log', 'a').close()

    print('Starting bot...')
    app = Application.builder().token(Token).build()

    app.add_handler(CommandHandler('start', start))
    app.add_handler(CommandHandler('help', help))
    app.add_handler(CommandHandler('info', info))
    app.add_handler(CommandHandler('predict', predict))
    app.add_handler(CommandHandler('report', get_report))

    app.add_error_handler(error)

    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.add_handler(MessageHandler(filters.TEXT, handle_text))

    print('Polling...')
    app.run_polling(poll_interval=3)