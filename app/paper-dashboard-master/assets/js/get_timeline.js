// Initialize the echarts instance based on the prepared dom
console.log('test')
async function postData(url='', data={}){
    const response = await fetch(url, {
        method : 'POST',
        mode: 'no-cors',
        headers : {
            'Content-Type' : 'application/json',
        },
        redirect: 'follow',
        body: JSON.stringify(data)
    }
        );
        console.log(data)
        return response.json();
            };
// console.log(data)
// console.log(today)

// setting for variable for request
// 요청 변수 설정 : pydantic에서 DtsInput에 해당하는 부분
console.log('test3')


const chat_rooms = document.getElementById('selchat').value;
const time_periods = document.getElementById('time_period').value;
const start_dates = document.getElementById('start_date').value;
console.log('test4')
console.log(start_dates)
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
    'chat_room' : 'KakaoTalk_Chat_IT개발자 구직_채용 정보교류방',
    // start_date : start_dates,
    'start_date' : '2023-01-22',
    'time_period' : time_periods,
    'penalty' : penalty_ls
};
console.log(request)
const url = 'http://127.0.0.1:30001';

const form = document.getElementById("my_form");
document.addEventListener("DOMContentLoaded", function() {
    // const form = document.getElementById("my_form");
    form.addEventListener("click", async function(event) {
        event.preventDefault();
        const result = await postData(url+'/dts' , request);
        // data{
        //  start : 시작 날짜임 str
        //  due : 마감 날짜임 str
        //  content : 한줄짜리 소개 markpoint 용
        //  dialouge : 원 본대화용 
        // }
        var data_for_x = []
        var series_data = result.map(x => 1)
        for (let i=0;i <result.length;i++){
            data_for_x.push(result[i]['start'])
        }
        var data_for_markpoint = []
        for (let i=0;i<result.length;i++){
            data_for_markpoint.push({name :i, value:result[i]['content'], xAxis:result[i]['start'], yAxis : 1, dialouge : result[i]['dialouge']})
        }

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
                data: ['Rainfall']
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
                name: 'Rainfall',
                type: 'line',
                data: series_data,
                markPoint: {
                    data: data_for_markpoint
                    // [
                    // // 데이터의 개수만큼
                    // // 포문 짜면되지않을까? -> 리스트 컴프리핸션 몰라 ㅋ
                    // { name: '0', value: '하윙', xAxis: 0, yAxis: 2 },
                    // { name: '1', value: '하윙', xAxis: 1, yAxis: 4.9 },
                    // { name: '2', value: '하윙', xAxis: 2, yAxis: 7 },
                    // { name: '3', value: '하윙', xAxis: 3, yAxis: 23.2 },
                    // { name: '4', value: '하윙', xAxis: 4, yAxis: 25.6 },
                    // { name: '5', value: '하윙', xAxis: 5, yAxis: 76.7 },
                    // { name: '6', value: '하윙', xAxis: 6, yAxis: 134.6 },
                    // { name: '7', value: '하윙', xAxis: 7, yAxis: 162.2 },
                    // { name: '8', value: '하윙', xAxis: 8, yAxis: 32.6 },
                    // { name: '9', value: '하윙', xAxis: 9, yAxis: 20 },
                    // { name: '10', value: '하윙', xAxis: 10, yAxis: 6.4 },
                    // { name: '11', value: '하윙', xAxis: 11, yAxis: 3.3 }
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
            if (params.componentType === 'markPoint'){
                var reques = { dialouge : params.dialouge};
                const data = postData(url+'/summary', reques);
                // data
                const summary = document.getElementById('summary')
                let ele = document.createElement('textarea')
                ele.style.weight = 700
                ele.style.height = 200
                ele.innerText(data)
                summary.appendChild(ele)
            }
            else if (params.componentType === 'series'){
            }
        })
    });
});