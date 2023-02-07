// async function postData(url = '', data = {}) {
//     response = await fetch(url, {
//       method: 'POST',
//       mode: 'no-cors',
//       headers: {
//         'Content-Type': 'application/json'
//       },
//       redirect: 'follow',
//       body: JSON.stringify(data)
//     });
//     result = await response.json()
//   }

// let query = {
//     items : 'hello'
// }
// let url = "http://localhost:30002"

// const form = document.getElementById("form");
// const chatting = document.getElementById('chatting')
// document.addEventListener("DOMContentLoaded", function() {
//     // const form = document.getElementById("my_form");
//     form.addEventListener("click", function(event) {
//         event.preventDefault();
//         console.log('true')
//         console.log(query)
//         postData(url+'/test' , query).then((result) =>{
//         var el = document.createElement("p");
//         el.textContent = result.json();
//         chatting.appendChild(el);
//     });
//     });
// });
// console.log(1)
// console.log(2)
// console.log(3)


// callback Hell example
// promise is Java Script object for asynchronous operation.
// state : pending -> fulfilled or rejected
// Producer vs Consumer

// 1. Producer : 요구할 때 바로 실행
// Promise 는 만들면 바로 실행된다. 
const promise = new Promise((resolve, reject) =>{
    // doing some heavy work
    // 데이터 읽고 모델 받아오기
    console.log('doing something...');
    setTImeout(() => {
        // resolve('ellie'); //성공적으로 네트워크에서 가공한 데이터를 resolve라는 콜백함수에서 처리한다.
        reject(new Error('no network'))
    },2000)
})

// 2.Consumers: then, catch, finally
promise.then((value) => {
    console.log(value)
})
.catch(error => {
    console.log(error);
})
.finally(() => {
    console.log('finally'); // 어찌됐든 결국 마무리 된다.
});

