# Ví dụ bộ lọc nhãn dòng (train, chọn ngẫu nhiên seed 0)

`>>` = dòng nhãn 1 được giữ, `x>` = dòng nhãn 1 bị gỡ nhãn (thành 0) bởi luật dòng rác. Hàm dài chỉ in 60 dòng đầu.

## Luật 1: `--drop_trivial_flaw` (10 hàm có ít nhất một nhãn bị gỡ)

### idx 17133 (stunnel) – gỡ 1/16 nhãn: dòng 23 (brace)

```c
>>    0 | void unbind_ports(void) {
      1 |     SERVICE_OPTIONS *opt;
      2 | 
      3 |     s_poll_init(fds, 1);
      4 | 
>>    5 |     CRYPTO_THREAD_write_lock(stunnel_locks[LOCK_SECTIONS]);
      6 | 
>>    7 |     opt=service_options.next;
>>    8 |     service_options.next=NULL;
>>    9 |     service_free(&service_options);
     10 | 
>>   11 |     while(opt) {
     12 |         unsigned i;
     13 |         s_log(LOG_DEBUG, "Unbinding service [%s]", opt->servname);
     14 |         for(i=0; i<opt->local_addr.num; ++i)
     15 |             unbind_port(opt, i);
     16 |         /* exec+connect service */
     17 |         if(opt->exec_name && opt->connect_addr.names) {
     18 |             /* create exec+connect services             */
     19 |             /* FIXME: this is just a crude workaround   */
     20 |             /*        is it better to kill the service? */
     21 |             /* FIXME: this won't work with FORK threads */
     22 |             opt->option.retry=0;
x>   23 |         }
     24 |         /* purge session cache of the old SSL_CTX object */
     25 |         /* this workaround won't be needed anymore after */
     26 |         /* delayed deallocation calls SSL_CTX_free()     */
>>   27 |         if(opt->ctx)
>>   28 |             SSL_CTX_flush_sessions(opt->ctx,
>>   29 |                 (long)time(NULL)+opt->session_timeout+1);
>>   30 |         s_log(LOG_DEBUG, "Service [%s] closed", opt->servname);
     31 | 
     32 |         {
>>   33 |             SERVICE_OPTIONS *garbage=opt;
>>   34 |             opt=opt->next;
>>   35 |             garbage->next=NULL;
>>   36 |             service_free(garbage);
     37 |         }
     38 |     }
     39 | 
>>   40 |     CRYPTO_THREAD_unlock(stunnel_locks[LOCK_SECTIONS]);
     41 | }
```

### idx 18299 (sudo) – gỡ 32/96 nhãn: dòng 1 (brace), dòng 5 (blank), dòng 6 (comment), dòng 11 (brace), dòng 15 (brace), dòng 17 (blank), dòng 20 (blank), dòng 21 (comment), dòng 26 (brace), dòng 27 (blank), dòng 28 (comment), dòng 32 (brace), dòng 42 (else), dòng 43 (comment), dòng 48 (brace), dòng 49 (brace), dòng 50 (brace), dòng 51 (blank), dòng 52 (comment), dòng 57 (blank), dòng 58 (comment), dòng 61 (blank), dòng 65 (brace), dòng 67 (blank), dòng 73 (blank), dòng 79 (blank), dòng 85 (brace), dòng 86 (blank), dòng 89 (brace), dòng 90 (blank), dòng 92 (blank), dòng 95 (brace)

```c
>>    0 | iolog_parse_json_object(struct json_object *object, struct eventlog *evlog)
x>    1 | {
>>    2 |     struct json_item *item;
>>    3 |     bool ret = false;
>>    4 |     debug_decl(iolog_parse_json_object, SUDO_DEBUG_UTIL);
x>    5 | 
x>    6 |     /* First object holds all the actual data. */
>>    7 |     item = TAILQ_FIRST(&object->items);
>>    8 |     if (item == NULL) {
>>    9 | 	sudo_warnx("%s", U_("missing JSON_OBJECT"));
>>   10 | 	goto done;
x>   11 |     }
>>   12 |     if (item->type != JSON_OBJECT) {
>>   13 | 	sudo_warnx(U_("expected JSON_OBJECT, got %d"), item->type);
>>   14 | 	goto done;
x>   15 |     }
>>   16 |     object = &item->u.child;
x>   17 | 
>>   18 |     TAILQ_FOREACH(item, &object->items, entries) {
>>   19 | 	struct iolog_json_key *key;
x>   20 | 
x>   21 | 	/* expecting key:value pairs */
>>   22 | 	if (item->name == NULL) {
>>   23 | 	    sudo_debug_printf(SUDO_DEBUG_WARN|SUDO_DEBUG_LINENO,
>>   24 | 		"%s: missing object name", __func__);
>>   25 | 	    goto done;
x>   26 | 	}
x>   27 | 
x>   28 | 	/* lookup name */
>>   29 | 	for (key = iolog_json_keys; key->name != NULL; key++) {
>>   30 | 	    if (strcmp(item->name, key->name) == 0)
>>   31 | 		break;
x>   32 | 	}
>>   33 | 	if (key->name == NULL) {
>>   34 | 	    sudo_debug_printf(SUDO_DEBUG_WARN|SUDO_DEBUG_LINENO,
>>   35 | 		"%s: unknown key %s", __func__, item->name);
>>   36 | 	} else if (key->type != item->type &&
>>   37 | 		(key->type != JSON_ID || item->type != JSON_NUMBER)) {
>>   38 | 	    sudo_debug_printf(SUDO_DEBUG_WARN|SUDO_DEBUG_LINENO,
>>   39 | 		"%s: key mismatch %s type %d, expected %d", __func__,
>>   40 | 		item->name, item->type, key->type);
>>   41 | 	    goto done;
x>   42 | 	} else {
x>   43 | 	    /* Matched name and type. */
>>   44 | 	    if (!key->setter(item, evlog)) {
>>   45 | 		sudo_debug_printf(SUDO_DEBUG_ERROR|SUDO_DEBUG_LINENO,
>>   46 | 		    "unable to store %s", key->name);
>>   47 | 		goto done;
x>   48 | 	    }
x>   49 | 	}
x>   50 |     }
x>   51 | 
x>   52 |     /* Merge cmd and argv as sudoreplay expects. */
>>   53 |     if (evlog->command != NULL && evlog->argv != NULL && evlog->argv[0] != NULL) {
>>   54 | 	size_t len, bufsize = strlen(evlog->command) + 1;
>>   55 | 	char *cp, *buf;
>>   56 | 	int ac;
x>   57 | 
x>   58 | 	/* Skip argv[0], we use evlog->command instead. */
>>   59 | 	for (ac = 1; evlog->argv[ac] != NULL; ac++)
   ... (36 more lines, 36 of them labelled)
```

### idx 1081 (linux) – gỡ 7/12 nhãn: dòng 5 (comment), dòng 6 (comment), dòng 7 (comment), dòng 8 (comment), dòng 9 (comment), dòng 10 (comment), dòng 11 (comment)

```c
>>    0 | static void touch_pmd(struct vm_area_struct *vma, unsigned long addr,
>>    1 | 		pmd_t *pmd)
      2 | {
      3 | 	pmd_t _pmd;
      4 | 
x>    5 | 	/*
x>    6 | 	 * We should set the dirty bit only for FOLL_WRITE but for now
x>    7 | 	 * the dirty bit in the pmd is meaningless.  And if the dirty
x>    8 | 	 * bit will become meaningful and we'll only set it with
x>    9 | 	 * FOLL_WRITE, an atomic set_bit will be required on the pmd to
x>   10 | 	 * set the young bit, instead of the current set_pmd_at.
x>   11 | 	 */
>>   12 | 	_pmd = pmd_mkyoung(pmd_mkdirty(*pmd));
>>   13 | 	if (pmdp_set_access_flags(vma, addr & HPAGE_PMD_MASK,
>>   14 | 				pmd, _pmd,  1))
     15 | 		update_mmu_cache_pmd(vma, addr, pmd);
     16 | }
     17 | 
```

### idx 9360 (src) – gỡ 2/5 nhãn: dòng 1 (brace), dòng 5 (brace)

```c
>>    0 | monitor_sync(struct monitor *pmonitor)
x>    1 | {
>>    2 | 	if (options.compression) {
      3 | 		/* The member allocation is not visible, so sync it */
>>    4 | 		mm_share_sync(&pmonitor->m_zlib, &pmonitor->m_zback);
x>    5 | 	}
      6 | }
```

### idx 21829 (Chrome) – gỡ 1/4 nhãn: dòng 4 (brace)

```c
      0 |   void DoCheckFakeData(uint8* audio_data, size_t length) {
>>    1 |     Type* output = reinterpret_cast<Type*>(audio_data);
>>    2 |     for (size_t i = 0; i < length; i++) {
>>    3 |       EXPECT_TRUE(algorithm_.is_muted() || output[i] != 0);
x>    4 |     }
      5 |    }
      6 | 
```

### idx 20723 (Chrome) – gỡ 1/10 nhãn: dòng 1 (brace)

```c
      0 | static inline RefPtr<DocumentFragment> createFragmentFromSource(const String& sourceString, const String& sourceMIMEType, Document* outputDoc)
x>    1 | {
>>    2 |     RefPtr<DocumentFragment> fragment = outputDoc->createDocumentFragment();
>>    3 |     if (sourceMIMEType == "text/html") {
>>    4 |         RefPtr<HTMLBodyElement> fakeBody = HTMLBodyElement::create(outputDoc);
>>    5 |         fragment->parseHTML(sourceString, fakeBody.get());
>>    6 |     } else if (sourceMIMEType == "text/plain")
>>    7 |         fragment->parserAddChild(Text::create(outputDoc, sourceString));
      8 |     else {
>>    9 |         bool successfulParse = fragment->parseXML(sourceString, 0);
>>   10 |         if (!successfulParse)
     11 |             return 0;
     12 |     }
>>   13 |     return fragment;
     14 | }
     15 | 
```

### idx 17806 (FORT-validator) – gỡ 2/5 nhãn: dòng 1 (brace), dòng 4 (brace)

```c
>>    0 | rdr_notification_ctx_cleanup(struct rdr_notification_ctx *ctx)
x>    1 | {
>>    2 | 	if (ctx->deltas.array != NULL)
>>    3 | 		deltas_parsed_cleanup(&ctx->deltas, __delta_head_destroy);
x>    4 | }
```

### idx 11482 (lftp) – gỡ 3/12 nhãn: dòng 1 (brace), dòng 46 (brace), dòng 47 (else)

```c
      0 | int   MirrorJob::Do()
x>    1 | {
      2 |    int	 res;
      3 |    int	 m=STALL;
      4 |    FileInfo *file;
      5 |    Job	 *j;
      6 | 
      7 |    switch(state)
      8 |    {
      9 |    case(INITIAL_STATE):
     10 |       remove_this_source_dir=(remove_source_dirs && source_dir.last_char()!='/');
     11 |       if(!strcmp(target_dir,".") || !strcmp(target_dir,"..") || (FlagSet(SCAN_ALL_FIRST) && parent_mirror))
     12 | 	 create_target_dir=false;
     13 | 
     14 |       source_session->Chdir(source_dir);
     15 |       source_redirections=0;
     16 |       source_session->Roll();
     17 |       set_state(CHANGING_DIR_SOURCE);
     18 |       m=MOVED;
     19 |       /*fallthrough*/
     20 |    case(CHANGING_DIR_SOURCE):
     21 |       HandleChdir(source_session,source_redirections);
     22 |       if(state!=CHANGING_DIR_SOURCE)
     23 | 	 return MOVED;
     24 |       if(source_session->IsOpen())
     25 | 	 return m;
     26 | 
     27 |       source_dir.set(source_session->GetCwd().GetDirectory());
     28 | 
     29 |    pre_MAKE_TARGET_DIR:
     30 |    {
     31 |       if(!create_target_dir)
     32 | 	 goto pre_CHANGING_DIR_TARGET;
     33 |       if(target_is_local)
     34 |       {
     35 | 	 struct stat st;
     36 | 	 if((FlagSet(RETR_SYMLINKS)?stat:lstat)(target_dir,&st)!=-1)
     37 | 	 {
     38 | 	    if(S_ISDIR(st.st_mode))
     39 | 	    {
     40 | 	       // try to enable write access
     41 | 	       // only if not enabled as chmod can clear sgid flags on directories
     42 | 	       if(!script_only && (st.st_mode!=(st.st_mode|0700)))
     43 | 		  chmod(target_dir,st.st_mode|0700);
     44 | 	       create_target_dir=false;
     45 | 	       goto pre_CHANGING_DIR_TARGET;
x>   46 | 	    }
x>   47 | 	    else
     48 | 	    {
     49 | 	       Report(_("Removing old local file `%s'"),target_dir.get());
     50 | 	       if(script)
     51 | 	       {
>>   52 | 		  ArgV args("rm");
     53 | 		  args.Append(target_session->GetFileURL(target_dir));
     54 | 		  xstring_ca cmd(args.CombineQuoted());
     55 | 		  fprintf(script,"%s\n",cmd.get());
     56 | 	       }
     57 | 	       if(!script_only)
     58 | 	       {
     59 | 		  if(remove(target_dir)==-1)
   ... (528 more lines, 8 of them labelled)
```

### idx 20627 (Chrome) – gỡ 1/4 nhãn: dòng 4 (brace)

```c
      0 | void Utterance::FinishAndDestroy() {
>>    1 |   completion_task_->Run();
>>    2 |   completion_task_ = NULL;
>>    3 |   delete this;
x>    4 | }
      5 | 
```

### idx 15344 (libredwg) – gỡ 3/20 nhãn: dòng 1 (brace), dòng 11 (brace), dòng 12 (else)

```c
      0 | dwg_free_object (Dwg_Object *obj)
x>    1 | {
      2 |   int error = 0;
      3 |   long unsigned int j;
      4 |   Dwg_Data *dwg;
      5 |   Bit_Chain *dat = &pdat;
      6 | 
      7 |   if (obj && obj->parent)
      8 |     {
>>    9 |       dwg = obj->parent;
     10 |       dat->version = dwg->header.version;
x>   11 |     }
x>   12 |   else
     13 |     return;
     14 |   if (obj->type == DWG_TYPE_FREED || obj->tio.object == NULL)
     15 |     return;
     16 |   dat->from_version = dat->version;
     17 |   if (obj->supertype == DWG_SUPERTYPE_UNKNOWN)
     18 |     goto unhandled;
     19 | 
     20 |   switch (obj->type)
     21 |     {
     22 |     case DWG_TYPE_TEXT:
     23 |       dwg_free_TEXT (dat, obj);
     24 |       break;
     25 |     case DWG_TYPE_ATTRIB:
     26 |       dwg_free_ATTRIB (dat, obj);
     27 |       break;
     28 |     case DWG_TYPE_ATTDEF:
     29 |       dwg_free_ATTDEF (dat, obj);
     30 |       break;
     31 |     case DWG_TYPE_BLOCK:
     32 |       dwg_free_BLOCK (dat, obj);
     33 |       break;
     34 |     case DWG_TYPE_ENDBLK:
     35 |       dwg_free_ENDBLK (dat, obj);
     36 |       break;
     37 |     case DWG_TYPE_SEQEND:
     38 |       dwg_free_SEQEND (dat, obj);
     39 |       break;
     40 |     case DWG_TYPE_INSERT:
     41 |       dwg_free_INSERT (dat, obj);
     42 |       break;
     43 |     case DWG_TYPE_MINSERT:
     44 |       dwg_free_MINSERT (dat, obj);
     45 |       break;
     46 |     case DWG_TYPE_VERTEX_2D:
     47 |       dwg_free_VERTEX_2D (dat, obj);
     48 |       break;
     49 |     case DWG_TYPE_VERTEX_3D:
     50 |       dwg_free_VERTEX_3D (dat, obj);
     51 |       break;
     52 |     case DWG_TYPE_VERTEX_MESH:
     53 |       dwg_free_VERTEX_MESH (dat, obj);
     54 |       break;
     55 |     case DWG_TYPE_VERTEX_PFACE:
     56 |       dwg_free_VERTEX_PFACE (dat, obj);
     57 |       break;
     58 |     case DWG_TYPE_VERTEX_PFACE_FACE:
     59 |       dwg_free_VERTEX_PFACE_FACE (dat, obj);
   ... (257 more lines, 16 of them labelled)
```

## Luật 2: `--max_flaw_density 0.5` (10 hàm bị bỏ cùng hàm cặp; tổng 600 hàm lỗi)

### idx 25378 (Android) – mật độ 0.57 (4/7 dòng), hàm cặp 25387

```c
      0 |    static void SetUpTestCase() {
>>    1 |     source_data_ = reinterpret_cast<uint8_t*>(
>>    2 |          vpx_memalign(kDataAlignment, kDataBlockSize));
>>    3 |     reference_data_ = reinterpret_cast<uint8_t*>(
>>    4 |          vpx_memalign(kDataAlignment, kDataBufferSize));
      5 |    }
      6 | 
```

### idx 13964 (tensorflow) – mật độ 1.00 (2/2 dòng), hàm cặp 13963

```c
>>    0 | TfDatazMetricsCollector::TfDatazMetricsCollector(const Env& env)
>>    1 |     : latency_estimator_(env) {}
```

### idx 22946 (Chrome) – mật độ 0.55 (17/31 dòng), hàm cặp 22952

```c
>>    0 |  void GM2TabStyle::PaintTabBackground(gfx::Canvas* canvas,
>>    1 |                                      bool active,
>>    2 |                                       int fill_id,
>>    3 |                                       int y_inset,
>>    4 |                                       const SkPath* clip) const {
      5 |    DCHECK(!y_inset || fill_id);
      6 |  
>>    7 |   const SkColor active_color =
>>    8 |       tab_->controller()->GetTabBackgroundColor(TAB_ACTIVE);
>>    9 |   const SkColor inactive_color =
>>   10 |       tab_->GetThemeProvider()->GetDisplayProperty(
>>   11 |           ThemeProperties::SHOULD_FILL_BACKGROUND_TAB_COLOR)
>>   12 |           ? tab_->controller()->GetTabBackgroundColor(TAB_INACTIVE)
>>   13 |           : SK_ColorTRANSPARENT;
     14 |    const SkColor stroke_color =
     15 |        tab_->controller()->GetToolbarTopSeparatorColor();
>>   16 |   const bool paint_hover_effect = !active && IsHoverActive();
>>   17 |   const float stroke_thickness = GetStrokeThickness(active);
     18 |  
>>   19 |   PaintTabBackgroundFill(canvas, active, paint_hover_effect, active_color,
>>   20 |                          inactive_color, fill_id, y_inset);
     21 |    if (stroke_thickness > 0) {
     22 |      gfx::ScopedCanvas scoped_canvas(clip ? canvas : nullptr);
     23 |      if (clip)
     24 |        canvas->sk_canvas()->clipPath(*clip, SkClipOp::kDifference, true);
>>   25 |     PaintBackgroundStroke(canvas, active, stroke_color);
     26 |    }
     27 |  
     28 |    PaintSeparators(canvas);
     29 |  }
     30 | 
```

### idx 8696 (git) – mật độ 0.57 (4/7 dòng), hàm cặp 8694

```c
>>    0 | static void record_recent_object(struct object *obj,
>>    1 | 				 struct strbuf *path,
>>    2 | 				 const char *last,
>>    3 | 				 void *data)
      4 | {
      5 | 	sha1_array_append(&recent_objects, obj->oid.hash);
      6 | }
```

### idx 17784 (FORT-validator) – mật độ 0.89 (8/9 dòng), hàm cặp None

```c
>>    0 | deltas_head_cleanup(struct deltas_head *list)
>>    1 | {
>>    2 | 	size_t i;
      3 | 
>>    4 | 	for (i = 0; i < list->capacity; i++)
>>    5 | 		delta_head_destroy(list->array[i]);
>>    6 | 	if (list->array)
>>    7 | 		free(list->array);
>>    8 | }
```

### idx 8718 (git) – mật độ 0.88 (7/8 dòng), hàm cặp None

```c
>>    0 | char *path_name(struct strbuf *path, const char *name)
      1 | {
>>    2 | 	struct strbuf ret = STRBUF_INIT;
>>    3 | 	if (path)
>>    4 | 		strbuf_addbuf(&ret, path);
>>    5 | 	strbuf_addstr(&ret, name);
>>    6 | 	return strbuf_detach(&ret, NULL);
>>    7 | }
```

### idx 5606 (linux) – mật độ 0.86 (6/7 dòng), hàm cặp 5605

```c
>>    0 | static void pcrypt_free(struct crypto_instance *inst)
>>    1 | {
>>    2 | 	struct pcrypt_instance_ctx *ctx = crypto_instance_ctx(inst);
      3 | 
>>    4 | 	crypto_drop_aead(&ctx->spawn);
>>    5 | 	kfree(inst);
>>    6 | }
```

### idx 15386 (frr) – mật độ 1.00 (195/195 dòng), hàm cặp 15379

```c
>>    0 | bgp_capability_parse (struct peer *peer, u_char *pnt, u_char length,
>>    1 | 		      u_char **error)
>>    2 | {
>>    3 |   int ret;
>>    4 |   u_char *end;
>>    5 |   struct capability cap;
>>    6 | 
>>    7 |   end = pnt + length;
>>    8 | 
>>    9 |   while (pnt < end)
>>   10 |     {
>>   11 |       afi_t afi;
>>   12 |       safi_t safi;
>>   13 | 
>>   14 |       /* Fetch structure to the byte stream. */
>>   15 |       memcpy (&cap, pnt, sizeof (struct capability));
>>   16 | 
>>   17 |       afi = ntohs(cap.mpc.afi);
>>   18 |       safi = cap.mpc.safi;
>>   19 | 
>>   20 |       if (BGP_DEBUG (normal, NORMAL))
>>   21 | 	zlog_debug ("%s OPEN has CAPABILITY code: %d, length %d",
>>   22 | 		   peer->host, cap.code, cap.length);
>>   23 | 
>>   24 |       /* We need at least capability code and capability length. */
>>   25 |       if (pnt + 2 > end)
>>   26 | 	{
>>   27 | 	  zlog_info ("%s Capability length error", peer->host);
>>   28 | 	  bgp_notify_send (peer, BGP_NOTIFY_CEASE, 0);
>>   29 | 	  return -1;
>>   30 | 	}
>>   31 | 
>>   32 |       /* Capability length check. */
>>   33 |       if (pnt + (cap.length + 2) > end)
>>   34 | 	{
>>   35 | 	  zlog_info ("%s Capability length error", peer->host);
>>   36 | 	  bgp_notify_send (peer, BGP_NOTIFY_CEASE, 0);
>>   37 | 	  return -1;
>>   38 | 	}
>>   39 | 
>>   40 |       /* We know MP Capability Code. */
>>   41 |       if (cap.code == CAPABILITY_CODE_MP)
>>   42 | 	{
>>   43 | 	  if (BGP_DEBUG (normal, NORMAL))
>>   44 | 	    zlog_debug ("%s OPEN has MP_EXT CAP for afi/safi: %u/%u",
>>   45 | 		       peer->host, afi, safi);
>>   46 | 
>>   47 | 	  /* Ignore capability when override-capability is set. */
>>   48 | 	  if (! CHECK_FLAG (peer->flags, PEER_FLAG_OVERRIDE_CAPABILITY))
>>   49 | 	    {
>>   50 | 	      /* Set negotiated value. */
>>   51 | 	      ret = bgp_capability_mp (peer, &cap);
>>   52 | 
>>   53 | 	      /* Unsupported Capability. */
>>   54 | 	      if (ret < 0)
>>   55 | 		{
>>   56 | 		  /* Store return data. */
>>   57 | 		  memcpy (*error, &cap, cap.length + 2);
>>   58 | 		  *error += cap.length + 2;
>>   59 | 		}
   ... (135 more lines, 135 of them labelled)
```

### idx 23456 (Chrome) – mật độ 0.57 (8/14 dòng), hàm cặp 23457

```c
>>    0 |   void UnwrapAndVerifyMojoHandle(mojo::ScopedSharedBufferHandle buffer_handle,
>>    1 |                                  size_t expected_size,
>>    2 |                                  bool expected_read_only_flag) {
      3 |      base::SharedMemoryHandle memory_handle;
      4 |      size_t memory_size = 0;
>>    5 |     bool read_only_flag = false;
>>    6 |     const MojoResult result =
>>    7 |         mojo::UnwrapSharedMemoryHandle(std::move(buffer_handle), &memory_handle,
>>    8 |                                        &memory_size, &read_only_flag);
      9 |      EXPECT_EQ(MOJO_RESULT_OK, result);
     10 |      EXPECT_EQ(expected_size, memory_size);
>>   11 |     EXPECT_EQ(expected_read_only_flag, read_only_flag);
     12 |    }
     13 | 
```

### idx 9300 (openjpeg) – mật độ 0.75 (6/8 dòng), hàm cặp None

```c
>>    0 | static unsigned short get_ushort(const unsigned char *data)
      1 | {
>>    2 |     unsigned short val = *(const unsigned short *)data;
>>    3 | #ifdef OPJ_BIG_ENDIAN
>>    4 |     val = ((val & 0xffU) << 8) | (val >> 8);
>>    5 | #endif
>>    6 |     return val;
      7 | }
```
