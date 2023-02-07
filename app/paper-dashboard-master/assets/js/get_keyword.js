// Initialize the echarts instance based on the prepared dom
// console.log('innit')

function get_Keyword(data){
    var data = data['result']
    var keyword = []
    var scores = []
    var data_for_series = []
    for (let i=0;i <5;i++){
        keyword.push(data[i][0]) //keyword
        scores.push(data[i][1]) //scores
        data_for_series.push({name : data[i][0], value : data[i][1]})
    }
    console.log('keyword',keyword)
    console.log('scores',scores)
    console.log('data_for_series',data_for_series)
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
};

// form.addEventListener("click", async function(event) {
//         event.preventDefault();
//         var chat_rooms3 = select.value;
//         var time_periods3 = document.getElementById('time_period').value;
//         var start_dates3 = document.getElementById('start_date').value;
//         console.log('test4')
//         // console.log(start_dates)
//         const b = document.getElementById('interest-Keyword').children;
//         var penalty_ls3 = [];
//         for (let i=0;i <= 16;i++){
//             if (b[`checkbox${i}`].checked){
//                 penalty_ls.push(b[`checkbox${i}`].value)
//             }
//         };
//         console.log('test3')
//         var request3 ={
//             chat_room : chat_rooms3,
//             // "chat_room" : "IT 개발자 구직 채용 정보교류방",
//             start_date : start_dates3,
//             // "start_date" : "3033-13-16",
//             time_period : time_periods3,
//             // "time_period" : "10",
//             penalty : penalty_ls3
//             // "penalty" :["AI","ML"]
//         };
//         await postData(url+'/keyword' , request3).then(get_Keyword);
// });
    
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
