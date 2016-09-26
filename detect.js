var fs = require('fs');
var readline = require('readline');
var cv = require('opencv');
var async = require("async");

var delay = 20;
var color = [0, 255, 0];
var trainImageCsv = "gender.csv";
var trainResultGender = "data/train.gender.fisher";
var trainResultAge = "data/train.age.lbph";
var imageWidth = 92;
var imageHeight = 112;
var downScale = 1; // 2 == 50% reduction in size
var faceCascade = './data/haar_face.xml';
// path for lighter cascade: './data/light_haar_face.xml'

var camera = new cv.VideoCapture(0);
var namedWindow = new cv.NamedWindow('Video', 0);
var fisherFaceRecognizer = cv.FaceRecognizer.createFisherFaceRecognizer();
fisherFaceRecognizer.loadSync(trainResultGender);
var lbphFaceRecognizer = cv.FaceRecognizer.createLBPHFaceRecognizer();
lbphFaceRecognizer.loadSync(trainResultAge);

/*
 * train should be done in another separate script
 * only load trainData here
 *
 * image directory structure:
 images/female/0.jpg
 images/female/1.jpg
 images/male/0.jpg
 images/male/1.jpg
 * csv:
 images/female/0.jpg;0
 images/female/1.jpg;0
 images/male/0.jpg;1
 images/male/1.jpg;1
 */

var train = function() {
  var trainData = [];
  readline.createInterface({
    input: fs.createReadStream(trainImageCsv),
    terminal: false
  }).on('line', function(line) {
    var arr = line.split(";");
    trainData.push([ parseInt(arr[1]), arr[0]]);
  }).on('close', function() {
    var fisherFacesRecognizer = cv.FaceRecognizer.createFisherFaceRecognizer();
    fisherFacesRecognizer.trainSync(trainData);
    fisherFacesRecognizer.saveSync(trainResultGender);
  });
};

var readCamera = function(cb) {
  camera.read(function(err, image) {
    if (err) {
      console.log(err);
    }
    if (image.width() > 0 && image.height() > 0) {
      image.resize(image.width() / downScale, image.height() / downScale);
      var original = image.copy();
      image.convertGrayscale();
      return cb(null, image, original);
    }
  });
};

var crop = function(im, face, width, height) {
  var ratio = 1;
  if (face.width > face.height) {
    ratio = width / face.width;
  } else {
    ratio = height / face.height;
  }

  var x = face.x - (((face.width  * ratio) - width  ) / 2 );
  var y = face.y - (((face.height * ratio) - height ) / 2 );

  var roi = im.roi(x, y, face.width, face.height);
  roi = roi.copy();
  roi.resize(width, height, cv.INTER_CUBIC);
  return roi;
};

var detectFaces = function(image, original, cb) {
  image.detectObject(faceCascade, {}, function(err, faces) {
    if (err) {
      console.log(err);
    }
    var len = faces.length;
    for (var i = 0; i < len; i++) {
      var face = faces[i];
      if(face.width <= imageWidth){
        continue;
      }
      var faceImg = crop(image, face, imageWidth, imageHeight);
      var predictionGender = fisherFaceRecognizer.predictSync(faceImg);
      var predictionAge = lbphFaceRecognizer.predictSync(faceImg);
      original.rectangle([face.x, face.y], [face.width, face.height], color);
      if (predictionGender.id === 0) {
        original.putText("Female: " + predictionAge.id, face.x + face.width / 3, face.y, null, color);
      } else {
        original.putText("Male: " + predictionAge.id, face.x + face.width / 3, face.y, null, color);
      }
      namedWindow.show(original);
      namedWindow.blockingWaitKey(0, 20);
    }
    return cb(null);
  });
};

async.forever(
  function(next) {
    async.waterfall([
      readCamera,
      detectFaces,
    ], function (err, result) {
      setTimeout(function() {
        next();
      }, delay);
    });
  },
  function(err) {
    console.error(err);
  }
);
