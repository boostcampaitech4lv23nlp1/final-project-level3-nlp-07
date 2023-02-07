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
    for (let i=0;i <10;i++){
        data_for_x.push(result[i]['start'])
        series_data.push(1)
    };
    console.log(series_data);
    var data_for_markpoint = []
    for (let i=0;i<10;i++){
        data_for_markpoint.push({name :i, value:result[i]['content'], xAxis:result[i]['start'], yAxis : 1, dialogue : result[i]['dialogue']})
    };

    var timeline_myChart = echarts.init(document.getElementById('charts'),null,{width : 800 , height : 400});
    
    var option = {
        title: {
            text: '주제로 나눠진 대화',
            left: 'center'
        },
        tooltip: {
            show : true,
            trigger: 'item',
            axisPointer: {
              type: 'cross'
            }
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
        dataZoom: [
            {
              type: 'inside',
              start: 50,
              end: 100
            },
            {
                show: true,
                type: 'slider',
                top: '90%',
                start: 50,
                end: 100
              }
            ],
        calculable: true,
        xAxis: {
            type: 'category',
            data: data_for_x,
            boundaryGap: false,
            axisLine: { onZero: false },
            splitLine: { show: false },
            min: 'dataMin',
            max: 'dataMax'
          },
        yAxis:
            {
            type: 'value'
            },
        series: [
            {
            name: 'DTS data',
            type: 'bar',
            data: series_data,
            markPoint: {
                data: data_for_markpoint
                // [
                // // 데이터의 개수만큼
                // // 포문 짜면되지않을까? -> 리스트 컴프리핸션 몰라 ㅋ
                // { name: '0', value: keyword or 한 문장, xAxis: start_date, yAxis: 1 },
                // ]
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
            let ele = document.createElement('h2')
            // ele.style.weight = 700;
            // ele.style.height = 200;
            ele.innerText(data);
            summary.appendChild(ele)
        })
        }
        else if (params.componentType === 'series'){
        }
        })
    };

function highlightchart(data){
    console.log('response data')
    console.log(data)
    console.log(typeof data)
    var for_data = JSON.parse(data)['timeline']
    console.log(for_data);
    var input_data = []
    for (let i = 0;i<20;i++){
        input_data.push({x : Date.parse(for_data[i]['start']),
        name : for_data[i]['content'],
        label : for_data[i]['content'],
        description : for_data[i]['dialogue'][0] + '...' + '요약 결과 보기',
        dialogue : for_data[i]['dialogue']
    })
    };
    console.log('input_data');
    // console.log(data[0]);
    // console.log(data[0]['start']);
    // console.log(data[0]['content']);
    console.log(input_data);
    Highcharts.chart('charts', {
        chart: {
            zoomType: 'x',
            type: 'timeline'
        },
        zooming :{
            type : "x"
        },
        xAxis: {
            type: 'datetime',
            visible: false,
            // minRange: 10 * 365 * 24 * 3600 * 1000
            minRange: 600 * 100
        },
    
        yAxis: {
            gridLineWidth: 1,
            visible: false,
            title: null,
            labels: {
                enabled: true
            }
        }, 
        legend: {
            enabled: true
        },
        title: {
            text: 'Timeline of OpenTalk'
        },
        subtitle: {
            text: 'Info source: <a href="https://en.wikipedia.org/wiki/Timeline_of_space_exploration">www.wikipedia.org</a>'
        },
        tooltip: {
          style: {
                width: 300
            }
        //   xDateFormat: '%A, %b %e, %H:%M:%S',
        //   dateTimeLabelFormats: {
        //     second: '%H:%M:%S'
        //   }
      },
        series: [{
            dataLabels: {
                allowOverlap: true,
                format: '<span style="color:{point.color}">● </span><span style="font-weight: bold;" > ' +
                    '{point.name}</span><br/>{point.x:%d %b %Y}'
                },
            events:{  // 이벤트
                click: function (event){ // 클릭이벤트 
                    console.log(event.point.dialogue);
                    var reques = { dialogue : event.point.dialogue};
                    alert('요약을 실행합니다....')
                    var rawdata =$('rawdata');
                    
                    postData(url+'/summary', reques).then(data => {
                    let ele = document.createElement('h4')
                    let text = document.createTextNode(data);
                    ele.appendChild(text);
                    summary.appendChild(ele)})
                    .catch((e) =>{
                        console.log(e)
                    })
                }
                },
            marker: {
                symbol: 'circle'
            },
            data: input_data
            }]
    });
    // 전체 대화수 녹이기
};

// console.log(data)
// console.log(today)

// setting for variable for request
// 요청 변수 설정 : pydantic에서 DtsInput에 해당하는 부분
console.log('test3');


// const chat_rooms = document.getElementById('selchat').value;
var chat_rooms = select.value;
var time_periods = document.getElementById('time_period').value;
var start_dates = document.getElementById('start_date').value;
console.log('test4')
// console.log(start_dates)
const b = document.getElementById('interest-Keyword').children;
var penalty_ls = [];
for (let i=0;i <= 16;i++){
    if (b[`checkbox${i}`].checked){
        penalty_ls.push(b[`checkbox${i}`].value)
    }
};
// console.log('test2')
// const request ={
//     chat_room : chat_rooms,
//     // "chat_room" : "IT 개발자 구직 채용 정보교류방",
//     start_date : start_dates,
//     // "start_date" : "2022-12-16",
//     time_period : time_periods,
//     // "time_period" : "10",
//     penalty : penalty_ls
//     // "penalty" :["AI","ML"]
// };
// console.log(request)

const form = document.getElementById("my_form");
form.addEventListener("click", function(event){
    event.preventDefault();
    console.log('request true');
    console.log(request);
    alert('채팅방 분석을 시작합니다..')
    var chat_rooms2 = select.value;
    var time_periods2 = document.getElementById('time_period').value;
    var start_dates2 = document.getElementById('start_date').value;
    console.log('test4')
    // console.log(start_dates)
    const b2 = document.getElementById('interest-Keyword').children;
    var penalty_ls2 = [];
    for (let i=0;i <= 16;i++){
        if (b2[`checkbox${i}`].checked){
            penalty_ls2.push(b2[`checkbox${i}`].value)
        }
    };
    console.log('test2')
    var request ={
        // "chat_room" : chat_rooms2,
        "chat_room" : "IT 개발자 구직 채용 정보교류방",
        // "start_date" : start_dates2,
        "start_date" : "2022-12-16",
        // "time_period" : time_periods2,
        "time_period" : "10",
        // "penalty" : penalty_ls2
        "penalty" :["AI","ML"]
    };
    console.log(request)
    postData(url+'/keywords' , request).then(get_Keyword).then((data) =>{
        alert('키워드 추출이 완료되었습니다. 대화 내 걸맞는 주제를 찾고 있습니다...')
    }
    );
    postData(url+'/dts' , request).then(highlightchart)
    .catch((error) =>{
        console.log(error)
        alert('입력이 거부되었습니다. DTSx')
    })
    .then((data) =>
        {
            alert('분석이 완료되었습니다.')
        }

    )
    .catch((data) => {
        console.log(data)
        alert('입력이 거부되었습니다. 다시 시도해주세요!')
    });
    }
)




