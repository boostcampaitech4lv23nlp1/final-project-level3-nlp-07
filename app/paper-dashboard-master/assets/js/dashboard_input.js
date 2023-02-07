let slider = document.querySelector(".slider");
let sliderValue = document.querySelector(".slider-value");
const url = 'http://127.0.0.1:30001';


$(document).ready(function(){
  $('ul.tabs li').click(function(){
      var tab_id = $(this).attr('data-tab');

      $('ul.tabs li').removeClass('current');
      $('.tab-content').removeClass('current');

      $(this).addClass('current');
      $("#"+tab_id).addClass('current');
  });
});


function postData(url = '', data = {}) {
  const response = fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json; charset=utf-8'
    },
    body: JSON.stringify(data)
  //   redirect : 'follow',
  //   referrerPolicy : 'same-origin'
  }).then((res) => {
      if (res.status === 200) {
          return res.json();
      } else if (res.status === 422) {
          console.log('여기가 입력이 뭘까 궁금한 부분')
          console.log(res)
          alret('분석 실패... 다시 시도해 주세요.')
      }
  }
  )
  return response;
};
slider.addEventListener("input", function() {
  sliderValue.innerHTML = this.value;
  });
//추후에 입력 받은 값으로 변경해야함
var options = ["Option 1", "Option 2", "Option 3", "Option 4"];

// 입력 받은 옵션을 기준으로 입력 옵션 생성
const select = document.getElementById("selchat");
let sel_chat = postData(url+'/chatlist', data ={"user_id" : "jaeuk", "password" : "1"})
.then((data) => {
  console.log(data)
    for (var i = 0; i < data['result'].length; i++) {
      console.log(data['result'][i])
      var opt = data['result'][i];
      var el = document.createElement("option");
      el.textContent = opt;
      el.value = opt;
      select.appendChild(el);
    };
  })


// 키값 생성 구조
const checkboxGroup = document.getElementById('interest-Keyword');
var data_for_keyword = ['채용', '취업', '코테', '인공지능', 'AI','알고리즘','면접','대기업', 'IT기업', 'IT', 'ML','DL','CNN', 'RNN', 'CV', 'NLP','Recsys'];

for (let i = 0; i < data_for_keyword.length; i++) {
  const checkbox = document.createElement('input');
  checkbox.setAttribute('type', 'checkbox');
  checkbox.setAttribute('id', `checkbox${i}`);
  checkbox.setAttribute('style', 'display: inline-block; margin-right: 10px; ');
  checkbox.setAttribute('value',data_for_keyword[i])

  const label = document.createElement('label');
  label.setAttribute('for',`checkbox${i}`);
  label.innerHTML = data_for_keyword[i];

  checkboxGroup.appendChild(checkbox);
  checkboxGroup.appendChild(label);
};

// 입력 날짜 최대 날짜 설정함.
const dateInput = document.getElementById("dateInput");
const today = new Date();
const todate = document.createElement('input');
todate.setAttribute('type', 'date');
todate.setAttribute('id', 'start_date');
todate.setAttribute('style', 'display: inline-block; width: 300px; height: 50px');
// Set the maximum date to today's date
todate.max = today.toISOString().split("T")[0];
dateInput.appendChild(todate)
