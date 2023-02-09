const mongoose = require('mongoose');
const { Schema } = mongoose;
 
// const { Types: { ObjectId } } = Schema; // ObjectId 타입은 따로 꺼내주어야 한다.
const chatSchema = new Schema({
  // _id 부분은 기본적으로 생략. 알아서 Object.id를 넣어줌
  user_id: {
    // type: ObjectId, // 몽고디비에서 ObjectId타입으로 데이터를 다룸
    type: String,
    required: true,
    ref: 'User', // user.js스키마에 reference로 연결되어 있음. join같은 기능. 나중에 populate에 사용
  },
  chat_id: {
    type: String,
    required: true,
  },
  createdAt: {
    type: Date,
    default: Date.now,
  },
});
 
module.exports = mongoose.model('chat', chatSchema);