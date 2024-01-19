import http from 'node:http';

const server = http.createServer();

server.on('request', (req, res) => {
  console.debug('request');

  res.statusCode = 200;

  return res.end();
});

server.listen(8080, 'localhost', () => {
  console.log('server started at http://localhost:8080');
});
