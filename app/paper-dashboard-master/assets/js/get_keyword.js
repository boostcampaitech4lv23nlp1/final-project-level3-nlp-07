// Initialize the echarts instance based on the prepared dom
console.log('innit')
async function postData(url='', data={}){
    const key_response = await fetch(url, {
        method : 'POST',
        headers : {
            'Content-Type' : 'application/json'
        },
        redirect: 'follow',
        body: JSON.stringify(data)
    }
        );
        // console.log(response)
        return key_response.json();
    };

const chat_room = document.getElementById('selchat').value;
const b2 = document.getElementById('interest-Keyword').children;
const penalty_ls2 = [];
for (let i=0;i <= 16;i++){
    if (b2[`checkbox${i}`].checked){
        penalty_ls2.push(b2[`checkbox${i}`].value)
    }
};

const request2 ={
    'chat_room' : chat_room,
    'penalty' : penalty_ls2
};

// const url = 'http://127.0.0.1:30001';

// const form = document.getElementById("my_form");
document.addEventListener("DOMContentLoaded", function() {
    form.addEventListener("click", async function(event) {
        event.preventDefault();
        const data = await postData(url+'/dts' , request2);
    // data{
    // List[List[str, int]]
    // }
        var keyword = []
        var scores = []
        var data_for_series = []
        for (let i=0;i <data.length;i++){
            keyword.push(data[i][0]) //keyword
            scores.push(data[i][1]) //scores
            data_for_series.push({name : data[i][0], value : data[i][1]})
        }
        var myChart = echarts.init(document.getElementById('keywords'),null);
        myChart.resize({
            width : "400%",
            height : "305%"
        });    
        var option = {
            tooltip: {
                trigger: 'item',
                formatter: '{a} <br/>{b} : {c} ({d}%)'
            },
            legend: {
                orient: 'vertical',
                // orient: 'horizontal',
                left: 'leftt',
                data: keyword
            },
            series: [
                {
                name: 'Access Source',
                type: 'pie',
                radius: '55%',
                center: ['50%', '60%'],
                data: data_for_series,
                emphasis: {
                    itemStyle: {
                    shadowBlur: 10,
                    shadowOffsetX: 0,
                    shadowColor: 'rgba(0, 0, 0, 0.5)'
                        }
                    }
                }
            ]
            };
            let currentIndex = -1;

        setInterval(function() {
        var dataLen = option.series[0].data.length;
        myChart.dispatchAction({
            type: 'downplay',
            seriesIndex: 0,
            dataIndex: currentIndex
        });
        currentIndex = (currentIndex + 1) % dataLen;
        myChart.dispatchAction({
            type: 'highlight',
            seriesIndex: 0,
            dataIndex: currentIndex
        });
        myChart.dispatchAction({
            type: 'showTip',
            seriesIndex: 0,
            dataIndex: currentIndex
        });
        }, 1000);

        // Display the chart using the configuration items and data just specified.
        myChart.setOption(option);
        myChart.on('click', function (params){
            if (params.componentType === 'markPoint'){
            window.open('https://search.naver.com/search.naver?sm=top_hty&fbm=1&ie=utf8&query=' +  encodeURIComponent(params.name))}
            else if (params.componentType === 'series'){

            }
        })
    });
});
    
// Specify the configuration items and data for the chart
// var option = {
//     tooltip: {
//         trigger: 'item',
//         formatter: '{a} <br/>{b} : {c} ({d}%)'
//     },
//     legend: {
//         orient: 'vertical',
//         // orient: 'horizontal',
//         left: 'leftt',
//         data: [
//         'AI',
//         'NLP',
//         '코테',
//         '취업',
//         'Recsys'
//         ]
//     },
//     series: [
//         {
//         name: 'Access Source',
//         type: 'pie',
//         radius: '55%',
//         center: ['50%', '60%'],
//         data: [
//             { value: 335, name: 'AI' },
//             { value: 310, name: 'NLP' },
//             { value: 234, name: '코테' },
//             { value: 135, name: '취업' },
//             { value: 1548, name: 'Recsys' }
//         ],
//         emphasis: {
//             itemStyle: {
//             shadowBlur: 10,
//             shadowOffsetX: 0,
//             shadowColor: 'rgba(0, 0, 0, 0.5)'
//             }
//         }
//         }
//     ]
//     };

//     let currentIndex = -1;

//     setInterval(function() {
//     var dataLen = option.series[0].data.length;
//     myChart.dispatchAction({
//         type: 'downplay',
//         seriesIndex: 0,
//         dataIndex: currentIndex
//     });
//     currentIndex = (currentIndex + 1) % dataLen;
//     myChart.dispatchAction({
//         type: 'highlight',
//         seriesIndex: 0,
//         dataIndex: currentIndex
//     });
//     myChart.dispatchAction({
//         type: 'showTip',
//         seriesIndex: 0,
//         dataIndex: currentIndex
//     });
//     }, 1000);

// // Display the chart using the configuration items and data just specified.
// myChart.setOption(option);
// myChart.on('click', function (params){
//     if (params.componentType === 'markPoint'){
//     window.open('https://search.naver.com/search.naver?sm=top_hty&fbm=1&ie=utf8&query=' +  encodeURIComponent(params.name))}
//     else if (params.componentType === 'series'){

//     }
// })
