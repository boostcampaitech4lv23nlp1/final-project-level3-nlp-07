let slider = document.querySelector(".slider");
let sliderValue = document.querySelector(".slider-value");

slider.addEventListener("input", function() {
  sliderValue.innerHTML = this.value;
  });
//추후에 입력 받은 값으로 변경해야함
var options = ["Option 1", "Option 2", "Option 3", "Option 4"];

// 입력 받은 옵션을 기준으로 입력 옵션 생성
const select = document.getElementById("selchat");

for (var i = 0; i < options.length; i++) {
  var opt = options[i];
  var el = document.createElement("option");
  el.textContent = opt;
  el.value = opt;
  select.appendChild(el);
};
console.log(select)


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
