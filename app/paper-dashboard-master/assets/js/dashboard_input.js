let slider = document.querySelector(".slider");
let sliderValue = document.querySelector(".slider-value");

slider.addEventListener("input", function() {
  sliderValue.innerHTML = this.value;
  });
var options = ["Option 1", "Option 2", "Option 3", "Option 4"];

var select = document.getElementById("selchat");

for (var i = 0; i < options.length; i++) {
  var opt = options[i];
  var el = document.createElement("option");
  el.textContent = opt;
  el.value = opt;
  select.appendChild(el);
}
console.log(select)

const checkboxGroup = document.getElementById('interest-Keyword');
var data = ['채용', '취업', '코테', '인공지능', 'AI','알고리즘','면접','대기업', 'IT기업', 'IT', 'ML','DL','CNN', 'RNN', 'CV', 'NLP','Recsys']

for (let i = 0; i < data.length; i++) {
  const checkbox = document.createElement('input');
  checkbox.setAttribute('type', 'checkbox');
  checkbox.setAttribute('id', `checkbox${i}`);
  checkbox.setAttribute('style', 'display: inline-block; margin-right: 10px; ');
  checkbox.setAttribute('value',data[i])

  const label = document.createElement('label');
  label.setAttribute('for',`checkbox${i}`);
  label.innerHTML = data[i];

  checkboxGroup.appendChild(checkbox);
  checkboxGroup.appendChild(label);
}
const dateInput = document.getElementById("dateInput");
const today = new Date();
const todate = document.createElement('input');
todate.setAttribute('type', 'date');
todate.setAttribute('id', 'start_date');
todate.setAttribute('style', 'display: inline-block; width: 300px; height: 50px');
// Set the maximum date to today's date
todate.max = today.toISOString().split("T")[0];
dateInput.appendChild(todate)
