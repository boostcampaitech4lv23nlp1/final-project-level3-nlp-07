const express = require('express');
// ...

// DB연결 모듈
const connect = require('./schemas');

// 몽고 디비 연결
connect(); 

// 라우터 모듈
// const indexRouter = require('./routes');
// const usersRouter = require('./routes/users');
// const commentsRouter = require('./routes/comments');

// ...

// 서버 연결
const app = express();
app.listen(30002, () => {
  console.log(app.get('port'), '번 포트에서 대기 중');
});