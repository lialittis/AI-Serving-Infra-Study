const {chromium}=require(process.env.P30_BROWSER_PACKAGE||'/tmp/inference-report-check/node_modules/playwright');
const fs=require('fs'),path=require('path');
const report=path.resolve(process.argv[2]||path.join(__dirname,'report'));
(async()=>{
 const browser=await chromium.launch({headless:true});
 const page=await browser.newPage({viewport:{width:1450,height:1000},offline:true});
 const errors=[];page.on('pageerror',e=>errors.push(String(e)));page.on('requestfailed',r=>errors.push(r.url()));
 await page.goto('file://'+path.join(report,'index.html'));
 await page.waitForFunction(()=>window.p30&&document.querySelector('#step').options.length===64);
 if(await page.locator('#examples button').count()!==4)throw Error('Missing four examples');
 for(const name of ['输入 H2D','线性层','RoPE','token D2H 与等待']){
   await page.getByRole('button',{name,exact:true}).click();
   if(await page.locator('#timeline .flow').count()<3)throw Error('Missing exact flow path');
   if(!/\/vllm-workspace\/|\/site-packages\/torch_npu\//.test(await page.locator('#source').textContent()))throw Error('Missing installed source');
 }
 await page.locator('#phases button').filter({hasText:/^调度$/}).click();
 if(!(await page.locator('#source').textContent()).includes('BalanceScheduler'))throw Error('Actual scheduler missing');
 const mode=await page.evaluate(()=>window.p30.summary.exact.mode);
 if(mode==='graph'){
   const proof=await page.evaluate(()=>{const d=window.p30,t=d.tasks.find(t=>t.replay_launch_id);
     return d.summary.graph.replay_by_step['32']>0 && d.views['32'].labels.filter(x=>x.startsWith('NPU stream')).length>1
       && t.queue_id===null && d.tasks.some(x=>x.id===t.replay_launch_id && x.cann_index!==null)});
   if(!proof)throw Error('Missing graph replay submission or separate stream rows');
   await page.getByRole('button',{name:'RoPE',exact:true}).click();
   if(!(await page.locator('#detail').textContent()).includes('图内 kernel 没有本次独立 launch'))throw Error('Invented per-kernel graph launch');
 }
 await page.locator('#timeline').scrollIntoViewIfNeeded();await page.screenshot({path:'/tmp/p30-timeline.png'});
 await page.selectOption('#step','0');if((await page.locator('#examples button').count())!==0)throw Error('decode examples on prefill');
 await page.selectOption('#step','63');await page.locator('#focus32').click();
 await page.setViewportSize({width:600,height:900});await page.locator('#timeline').scrollIntoViewIfNeeded();
 await page.screenshot({path:'/tmp/p30-narrow.png'});
 await page.goto('file://'+path.join(report,'decode32.svg'));
 if(await page.locator('rect title').count()<100)throw Error('Missing measured bars');
 if(errors.length)throw Error(errors.join('\n'));
 const out=path.join(__dirname,'results/validation');fs.mkdirSync(out,{recursive:true});
 fs.writeFileSync(path.join(out,'browser-'+path.basename(report)+'.json'),JSON.stringify({mode,offline:true,steps:64,examples:4,
   source_excerpts:true,actual_scheduler:true,exact_flow_highlighting:true,narrow_view:true,standalone_svg:true,errors},null,2)+'\n');
 console.log('Offline report checks passed');await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
