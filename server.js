//server.js
// const express = require('express');
// const favicon = require('express-favicon');
// const path = require('path');
// const port = process.env.PORT || 8080;
// const app = express();
// app.use(favicon(__dirname + '/build/favicon.ico'));
// // the __dirname is the current directory from where the script is running
// app.use(express.static(__dirname));
// app.use(express.static(path.join(__dirname, 'build')));
// // app.get('/ping', function (req, res) {
// //  return res.send('pong');
// // });
// app.get('/*', function (req, res) {
//   res.sendFile(path.join(__dirname, 'build', 'index.html'));
// });
// app.listen(port);


if (process.env.NODE_ENV === 'production') {
  // Exprees will serve up production assets
  app.use(express.static(path.join(__dirname, 'build')));

  // Express serve up index.html file if it doesn't recognize route
  const path = require('path');
  app.get('*', (req, res) => {
    res.sendFile(path.resolve(__dirname, 'build', 'index.html'));
  });
}