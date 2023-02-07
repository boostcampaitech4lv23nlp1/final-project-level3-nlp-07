// Initialize the echarts instance based on the prepared dom
console.log('test');
const summary = document.getElementById('summary');
function timeline(result){
    // result 구조
    // start : 시작날짜
    // content : 첫 문장 or 핵심 키워드 3개
    // 
    console.log('result');
    console.log(result);
    // 데이터 형 변환 중  
    // data_for_x  : xAxis를 위한 데이터
    // series_data : series를 위한 데이터 
    let data_for_x = [];
    let series_data = [];
    for (let i=0;i <result.length;i++){
        data_for_x.push(result[i]['start'])
        series_data.push(1)
    };
    console.log(series_data);
    var data_for_markpoint = []
    for (let i=0;i<result.length;i++){
        data_for_markpoint.push({name :i, value:result[i]['content'], xAxis:result[i]['start'], yAxis : 1, dialogue : result[i]['dialogue']})
    };

    var timeline_myChart = echarts.init(document.getElementById('charts'),null,{width : 800 , height : 400});
    
    var option = {
        title: {
            text: 'Rainfall',
            subtext: 'Fake Data'
        },
        tooltip: {
            trigger: 'axis'
        },
        legend: {
            data: ['분절된 대화']
        },
        toolbox: {
            show: true,
            feature: {
                dataView: { show: true, readOnly: false },
                magicType: { show: true, type: ['line', 'bar'] },
                restore: { show: true },
                saveAsImage: { show: true }
            }
        },
        dataZoom : {
            type : 'slider',
            show : true
        },
        calculable: true,
        xAxis: [
            {
            type: 'time',
            // type: 'category',
            // prettier-ignore
            data: data_for_x
            }
        ],
        yAxis: [
            {
            type: 'value'
            }
        ],
        series: [
            {
            name: 'DTS data',
            type: 'line',
            data: series_data,
            markPoint: {
                data: data_for_markpoint
                // [
                // // 데이터의 개수만큼
                // // 포문 짜면되지않을까? -> 리스트 컴프리핸션 몰라 ㅋ
                // { name: '0', value: keyword or 한 문장, xAxis: start_date, yAxis: 1 },
                // ]
            },
            markLine: {
                data: [{ yAxis: 1, name: 'threshold' }]
            }
            }
        ]
    };

    // Display the chart using the configuration items and data just specified.
    timeline_myChart.setOption(option);
    timeline_myChart.on('click', function (params){
        // markpoint만 설정했을 때 이와 같은 함쿼리 날리기 
        if (params.componentType === 'markPoint'){
            var reques = { dialogue : params.dialogue};
            postData(url+'/summary', reques).then(data => {
            let ele = document.createElement('textarea')
            ele.style.weight = 700;
            ele.style.height = 200;
            ele.innerText(data.json());
            summary.appendChild(ele)
        })
        }
        else if (params.componentType === 'series'){
        }
        })
    };

function postData(url = '', data = {}) {
    const response = fetch(url, {
      method: 'POST',
      mode: 'no-cors',
      credentials: 'include' ,
      cache : 'no-cache',
      headers: {
        // 'Content-Type': 'application/json; charset=UTF-8'
        'Content-Type': 'application/json'
        // 'Access-Control-Allow-Origin': ['*']
      },
      body: JSON.stringify(data)
    //   redirect : 'follow',
    //   referrerPolicy : 'same-origin'
    }).then((res) => {
        if (res.status === 200) {
            alert('대화방 분석 완료');
            return res.json();
        } else if (res.status === 422) {
            console.log('여기가 입력이 뭘까 궁금한 부분')
            console.log(res)
            alret('분석 실패... 다시 시도해 주세요.')
        }
    }
    )
    return response;
  }
// console.log(data)
// console.log(today)

// setting for variable for request
// 요청 변수 설정 : pydantic에서 DtsInput에 해당하는 부분
console.log('test3')


// const chat_rooms = document.getElementById('selchat').value;
const chat_rooms = select.value;
const time_periods = document.getElementById('time_period').value;
const start_dates = document.getElementById('start_date').value;
console.log('test4')
// console.log(start_dates)
const b = document.getElementById('interest-Keyword').children;
const penalty_ls = [];
for (let i=0;i <= 16;i++){
    if (b[`checkbox${i}`].checked){
        penalty_ls.push(b[`checkbox${i}`].value)
    }
};
console.log('test2')
const request ={
    // 'chat_room' : chat_room,
    "chat_room" : "IT 개발자 구직 채용 정보교류방",
    // start_date : start_dates,
    "start_date" : "2022-12-16",
    // time_period : time_period
    "time_period" : "10",
    // penalty : penalty
    "penalty" :["AI","ML"]
};
// console.log(request)
const url = 'http://101.101.218.23:30001';

const form = document.getElementById("my_form");
form.addEventListener("click", function(event){
    event.preventDefault();
    console.log('request true');
    console.log(request);
    postData(url+'/simple' , request).then((data) => {
        console.log(data);
        timeline(data.json());
    })
    .catch((data) => {
        console.log(data)
        alert('입력이 거부되었습니다.')
    });
    }
)




