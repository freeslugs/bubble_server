import os 
import time
from flask_socketio import SocketIO, emit, send
from flask import Flask, url_for, render_template, request, session, redirect, send_from_directory, send_file, jsonify
from cv_lib import get_cards
import serial
# ser = serial.Serial('/dev/tty.usbserial', 9600)

app = Flask(__name__, static_url_path='', static_folder='static')
app.debug = True
# socketio = SocketIO(app)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/post_image', methods=['POST'])
def post_image():
  file = request.files['file']
  # timestamp is name of image
  ts = int(time.time())
  filename = str(ts) + '.png'
  print("saving file")
  main_dir = os.getcwd()
  path = os.path.join(main_dir, "static", filename)
  file.save(path)
  cards = get_cards(path)
  print len(cards)
  return render_template('cards.html', cards=cards)

# @socketio.on('move slider')
# def handle_move_slider():
#   print "handle move slider"
#   done = move_slider()
#   if(done is True):
#     emit('processing cards', {'no': 'data'}, broadcast=True)
#     print "process and stich images"
#     print "find cards"
#     print ""
#   else: 
#     emit('get photo', {'no': 'data'}, broadcast=True)

# def move_slider():
#   print "moving the slider"
#   time.sleep(2)
#   return False

def hardware_move(a, b):
  print "moving from " + str(a) + " to " + str(b)
  # generate string 
  s = "MOVE " + str(a) +  " " + str(b)
  # ser.write(s)

def bubble_sort(alist):
  total = len(alist)
  for passnum in range(len(alist)-1,0,-1):
    for i in range(passnum):
      if alist[i]>alist[i+1]:
        hardware_move(total-i, 0)
        temp = alist[i]
        hardware_move(total-i+1, total-i)
        alist[i] = alist[i+1]
        hardware_move(0, total-i+1)
        alist[i+1] = temp


@app.route('/bubble_sort', methods=['POST'])
def handle_bubble_sort():
  l = [None] * len(request.form)
  print len(request.form)
  print request.form
  for card in request.form:
    print card
    print request.form[card]
    l[int(card)] = request.form[card]
  print l
  bubble_sort(l)
  return "All done"
  
if __name__ == "__main__":
  if not os.path.exists("temp"):
    os.makedirs("temp")  
  # socketio.run(app)
  app.run()

