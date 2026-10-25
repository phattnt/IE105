# Patch taxonomy examples (PrimeVul test, random per class, seed 0)

### needs_outside | util-linux idx 208140 | kind add_only | B ['other'] | outside ['callee']

```diff
--- 
+++ 
@@ -291,4 +291,7 @@
 		usage(stderr);
 
+	if (mnt_context_is_restricted(cxt))
+		sanitize_paths(cxt);
+
 	if (oper) {
 		/* MS_PROPAGATION operations, let's set the mount flags */
```

### needs_outside | dpdk idx 210283 | kind add_only | B ['other'] | outside ['callee', 'local', 'param']

```diff
--- 
+++ 
@@ -11,4 +11,7 @@
 	int fd, i;
 	int numa_node = SOCKET_ID_ANY;
+
+	if (validate_msg_fds(dev, ctx, 1) != 0)
+		return RTE_VHOST_MSG_RESULT_ERR;
 
 	fd = ctx->fds[0];
```

### needs_outside | pcre2 idx 195264 | kind modify | B ['other'] | outside ['macro', 'param']

```diff
--- 
+++ 
@@ -59,5 +59,5 @@
     SLJIT_ASSERT(*cc == XCL_PROP || *cc == XCL_NOTPROP);
     cc++;
-    if (*cc == PT_CLIST && *cc == XCL_PROP)
+    if (*cc == PT_CLIST && cc[-1] == XCL_PROP)
       {
       other_cases = PRIV(ucd_caseless_sets) + cc[1];
```

### needs_outside | hexchat idx 199767 | kind modify | B ['bounds_check'] | outside ['callee', 'local']

```diff
--- 
+++ 
@@ -3,6 +3,6 @@
 {
 	char buffer[256];	/* buffer for requesting capabilities and emitting the signal */
-	guint32 want_cap; /* format the CAP REQ string based on previous capabilities being requested or not */
-	guint32 want_sasl; /* CAP END shouldn't be sent when SASL is requested, it needs further responses */
+	gboolean want_cap = FALSE; /* format the CAP REQ string based on previous capabilities being requested or not */
+	gboolean want_sasl = FALSE; /* CAP END shouldn't be sent when SASL is requested, it needs further responses */
 	char **extensions;
 	int i;
@@ -10,6 +10,4 @@
 	EMIT_SIGNAL_TIMESTAMP (XP_TE_CAPLIST, serv->server_session, nick,
 								  extensions_str, NULL, NULL, 0, tags_data->timestamp);
-	want_cap = 0;
-	want_sasl = 0;
 
 	extensions = g_strsplit (extensions_str, " ", 0);
@@ -20,64 +18,25 @@
 	{
 		const char *extension = extensions[i];
+		gsize x;
 
-		if (!strcmp (extension, "identify-msg"))
+		/* if the SASL password is set AND auth mode is set to SASL, request SASL auth */
+		if (!g_strcmp0 (extension, "sasl") &&
+			((serv->loginmethod == LOGIN_SASL && strlen (serv->password) != 0)
+				|| (serv->loginmethod == LOGIN_SASLEXTERNAL && serv->have_cert)))
 		{
-			strcat (buffer, "identify-msg ");
-			want_cap = 1;
-		}
-		if (!strcmp (extension, "multi-prefix"))
-		{
-			strcat (buffer, "multi-prefix ");
-			want_cap = 1;
-		}
-		if (!strcmp (extension, "away-notify"))
-		{
-			strcat (buffer, "away-notify ");
-			want_cap = 1;
-		}
-		if (!strcmp (extension, "account-notify"))
-		{
-			strcat (buffer, "account-notify ");
-			want_cap = 1;
-		}
-		if (!strcmp (extension, "extended-join"))
-		{
-			strcat (buffer, "extended-join ");
-			want_cap = 1;
-		}
-		if (!strcmp (extension, "userhost-in-names"))
-		{
-			strcat (buffer, "userhost-in-names ");
-			want_cap = 1;
+			want_cap = TRUE;
+			want_sasl = TRUE;
+			g_strlcat (buffer, "sasl ", sizeof(buffer));
+			continue;
 		}
 
-		/* bouncers can prefix a name space to the extension so we should use.
-		 * znc <= 1.0 uses "znc.in/server-time" and newer use "znc.in/server-time-iso".
-		 */
-		if (!strcmp (extension, "znc.in/server-time-iso"))
+		for (x = 0; x < G_N_ELEMENTS(supported_caps); ++x)
 		{
-			strcat (buffer, "znc.in/server-time-iso ");
-			want_cap = 1;
-		}
-		if (!strcmp (extension, "znc.in/server-time"))
-		{
-			strcat (buffer, "znc.in/server-time ");
-			want_cap = 1;
-		}
-		if (prefs.hex_irc_cap_server_time
-			 && !strcmp (extension, "server-time"))
-		{
-			strcat (buffer, "server-time ");
-			want_cap = 1;
-		}
-		
-		/* if the SASL password is set AND auth mode is set to SASL, request SASL auth */
-		if (!strcmp (extension, "sasl")
-			&& ((serv->loginmethod == LOGIN_SASL && strlen (serv->password) != 0)
-			|| (serv->loginmethod == LOGIN_SASLEXTERNAL && serv->have_cert)))
-		{
-			strcat (buffer, "sasl ");
-			want_cap = 1;
-			want_sasl = 1;
+			if (!g_strcmp0 (extension, supported_caps[x]))
+			{
+				g_strlcat (buffer, extension, si
```

### needs_outside | radare2 idx 214282 | kind modify | B ['error_path', 'null_check'] | outside ['local', 'param']

```diff
--- 
+++ 
@@ -7,4 +7,8 @@
 	attr = r_bin_java_default_attr_new (bin, buffer, sz, buf_offset);
 	offset += 6;
+	if (buf_offset + offset + 8 > sz) {
+		eprintf ("Invalid amount of inner classes\n");
+		return NULL;
+	}
 	if (attr == NULL) {
 		// TODO eprintf
@@ -17,5 +21,5 @@
 	for (i = 0; i < attr->info.inner_classes_attr.number_of_classes; i++) {
 		curpos = buf_offset + offset;
-		if (offset + 8 > sz) {
+		if (buf_offset + offset + 8 > sz) {
 			eprintf ("Invalid amount of inner classes\n");
 			break;
```

### needs_outside | varnish-cache idx 212407 | kind add_only | B ['other'] | outside ['param']

```diff
--- 
+++ 
@@ -6,4 +6,6 @@
 	if (fm->hdf[u] & HDF_FILTER)
 		return (1);
+	if (u < HTTP_HDR_FIRST)
+		return (0);
 	e = strchr(fm->hd[u].b, ':');
 	if (e == NULL)
```

### needs_outside | linux idx 209955 | kind modify | B ['error_path', 'null_check', 'release'] | outside ['callee']

```diff
--- 
+++ 
@@ -29,4 +29,13 @@
 	mutex_init(&trans_pcie->mutex);
 	init_waitqueue_head(&trans_pcie->ucode_write_waitq);
+
+	trans_pcie->rba.alloc_wq = alloc_workqueue("rb_allocator",
+						   WQ_HIGHPRI | WQ_UNBOUND, 1);
+	if (!trans_pcie->rba.alloc_wq) {
+		ret = -ENOMEM;
+		goto out_free_trans;
+	}
+	INIT_WORK(&trans_pcie->rba.rx_alloc, iwl_pcie_rx_allocator_work);
+
 	trans_pcie->tso_hdr_page = alloc_percpu(struct iwl_tso_hdr_page);
 	if (!trans_pcie->tso_hdr_page) {
@@ -163,8 +172,4 @@
 	 }
 
-	trans_pcie->rba.alloc_wq = alloc_workqueue("rb_allocator",
-						   WQ_HIGHPRI | WQ_UNBOUND, 1);
-	INIT_WORK(&trans_pcie->rba.rx_alloc, iwl_pcie_rx_allocator_work);
-
 #ifdef CONFIG_IWLWIFI_DEBUGFS
 	trans_pcie->fw_mon_data.state = IWL_FW_MON_DBGFS_STATE_CLOSED;
@@ -178,4 +183,6 @@
 out_no_pci:
 	free_percpu(trans_pcie->tso_hdr_page);
+	destroy_workqueue(trans_pcie->rba.alloc_wq);
+out_free_trans:
 	iwl_trans_free(trans);
 	return ERR_PTR(ret);
```

### needs_outside | ImageMagick idx 202748 | kind add_only | B ['bounds_check'] | outside ['callee', 'macro']

```diff
--- 
+++ 
@@ -128,4 +128,6 @@
           one=1;
           image->colors=one << tga_info.bits_per_pixel;
+          if (image->colors > ((~0UL)/sizeof(*image->colormap)))
+            ThrowReaderException(CorruptImageError,"ImproperImageHeader");
           if (AcquireImageColormap(image,image->colors,exception) == MagickFalse)
             ThrowReaderException(ResourceLimitError,"MemoryAllocationFailed");
```

### a_to_b_local_field_macro | vim idx 199851 | kind add_only | B ['bounds_check'] | outside ['local']

```diff
--- 
+++ 
@@ -154,4 +154,9 @@
 		break;
 	    vcol += chartabsize(ptr + col, (colnr_T)vcol);
+	    if (vcol >= MAXCOL)
+	    {
+		emsg(_(e_resulting_text_too_long));
+		break;
+	    }
 	    if (has_mbyte)
 		col += (*mb_ptr2len)(ptr + col);
```

### a_to_b_local_field_macro | linux idx 197329 | kind modify | B ['other'] | outside ['local']

```diff
--- 
+++ 
@@ -63,5 +63,5 @@
 	rcu_assign_pointer(*fp, f);
 
-	if (fold && fold->handle && f->handle != fold->handle) {
+	if (fold) {
 		th = to_hash(fold->handle);
 		h = from_hash(fold->handle >> 16);
```

### a_to_b_local_field_macro | admesh idx 204534 | kind modify | B ['other'] | outside ['local']

```diff
--- 
+++ 
@@ -46,11 +46,10 @@
   neighbor2 = stl->neighbors_start[facet].neighbor[edge2];
 
-  if(neighbor1 == -1) {
+  if(neighbor1 == -1 && neighbor2 != -1) {
     stl_update_connects_remove_1(stl, neighbor2);
   }
-  if(neighbor2 == -1) {
+  else if (neighbor2 == -1 && neighbor1 != -1) {
     stl_update_connects_remove_1(stl, neighbor1);
   }
-
 
   neighbor3 = stl->neighbors_start[facet].neighbor[edge3];
```

### a_to_b_local_field_macro | gpac idx 195984 | kind modify | B ['bounds_check', 'null_check'] | outside ['field']

```diff
--- 
+++ 
@@ -1,18 +1,10 @@
 GF_Err diST_box_read(GF_Box *s, GF_BitStream *bs)
 {
-	u32 i;
-	char str[1024];
 	GF_DIMSScriptTypesBox *p = (GF_DIMSScriptTypesBox *)s;
 
-	i=0;
-	str[0]=0;
-	while (1) {
-		str[i] = gf_bs_read_u8(bs);
-		if (!str[i]) break;
-		i++;
-	}
-	ISOM_DECREASE_SIZE(p, i);
-
-	p->content_script_types = gf_strdup(str);
+	p->content_script_types = gf_malloc(sizeof(char) * (s->size+1));
+	if (!p->content_script_types) return GF_OUT_OF_MEM;
+	gf_bs_read_data(bs, p->content_script_types, s->size);
+	p->content_script_types[s->size] = 0;
 	return GF_OK;
 }
```

### a_to_b_local_field_macro | vim idx 200323 | kind modify | B ['bounds_check'] | outside ['field', 'macro']

```diff
--- 
+++ 
@@ -715,5 +715,6 @@
 						       ? DIFF_YES : DIFF_NONE;
 			}
-			else if (sp->ts_isdiff == DIFF_INSERT)
+			else if (sp->ts_isdiff == DIFF_INSERT
+							    && sp->ts_fidx > 0)
 			    // When inserting trail bytes don't advance in the
 			    // bad word.
```

### a_to_b_local_field_macro | libjpeg idx 195740 | kind modify | B ['error_path', 'null_check'] | outside ['local']

```diff
--- 
+++ 
@@ -18,7 +18,10 @@
     lines = 8;
   }
+
+  if (lines == 0)
+    return false;
+  
   if (m_pFrame->HeightOf() > 0)
     m_ulRemaining[0] -= lines;
-  assert(lines > 0);
   assert(m_ucCount < 4);
 
```

### a_to_b_local_field_macro | ImageMagick idx 197068 | kind modify | B ['other'] | outside ['local']

```diff
--- 
+++ 
@@ -221,15 +221,15 @@
             onto the stack, so we can restore them at the end of the sequence.
           */
-          info_copy=(DCMInfo *) AcquireMagickMemory(sizeof(info));
-          if (info_copy == (DCMInfo *) NULL)
+          DCMInfo *clone_info = (DCMInfo *) AcquireMagickMemory(sizeof(info));
+          if (clone_info == (DCMInfo *) NULL)
             ThrowDCMException(ResourceLimitError,"MemoryAllocationFailed")
-          (void) memcpy(info_copy,&info,sizeof(info));
-          info_copy->scale=(Quantum *) AcquireQuantumMemory(
-            info_copy->scale_size+1,sizeof(*info_copy->scale));
-          if (info_copy->scale == (Quantum *) NULL)
+          (void) memcpy(clone_info,&info,sizeof(info));
+          clone_info->scale=(Quantum *) AcquireQuantumMemory(
+            clone_info->scale_size+1,sizeof(*clone_info->scale));
+          if (clone_info->scale == (Quantum *) NULL)
             ThrowDCMException(ResourceLimitError,"MemoryAllocationFailed")
-          (void) memcpy(info_copy->scale,info.scale,info_copy->scale_size*
-            sizeof(*info_copy->scale));
-          AppendValueToLinkedList(stack,info_copy);
+          (void) memcpy(clone_info->scale,info.scale,clone_info->scale_size*
+            sizeof(*clone_info->scale));
+          AppendValueToLinkedList(stack,clone_info);
           sequence_depth++;
         }
```

### a_to_b_local_field_macro | chafa idx 214124 | kind add_only | B ['bounds_check', 'overflow'] | outside ['local', 'macro']

```diff
--- 
+++ 
@@ -28,4 +28,8 @@
 	} else if (code_new > current_entry) {
 		/* Code is invalid */
+		return LZW_BAD_CODE;
+
+	} else if (code_new >= 1 << LZW_CODE_MAX) {
+		/* Don't access out of bound */
 		return LZW_BAD_CODE;
 
```

### not_a_to_b | monit idx 217176 | kind modify | B ['other'] | outside []

```diff
--- 
+++ 
@@ -13,5 +13,5 @@
                         while ((n = fread(buf, sizeof(char), sizeof(buf) - 1, f)) > 0) {
                                 buf[n] = 0;
-                                StringBuffer_append(res->outputbuffer, "%s", buf);
+                                escapeHTML(res->outputbuffer, buf);
                         }
                         fclose(f);
```

### not_a_to_b | pjproject idx 195670 | kind add_only | B ['other'] | outside []

```diff
--- 
+++ 
@@ -98,4 +98,6 @@
 	while (pj_scan_strcmp(scanner, "]]>", 3)) {
 	    pj_str_t dummy;
+
+	    pj_scan_advance_n(scanner, 1, PJ_FALSE);
 	    pj_scan_get_until_ch(scanner, ']', &dummy);
 	}
```

### not_a_to_b | qemu idx 210887 | kind modify | B ['other'] | outside []

```diff
--- 
+++ 
@@ -6,5 +6,5 @@
     NetClientState *nc = qemu_get_queue(s->nic);
     if (s->phy_reg[PHY_CTRL] & MII_CR_LOOPBACK) {
-        nc->info->receive(nc, buf, size);
+        qemu_receive_packet(nc, buf, size);
     } else {
         qemu_send_packet(nc, buf, size);
```

### not_a_to_b | zsh idx 197850 | kind add_only | B ['init'] | outside []

```diff
--- 
+++ 
@@ -6,4 +6,10 @@
 	if ((ep = strchr(bv->fm, '}'))) {
 	    char oc = *ep, *col, *coll;
+	    int ops = opts[PROMPTSUBST], opb = opts[PROMPTBANG];
+	    int opp = opts[PROMPTPERCENT];
+
+	    opts[PROMPTPERCENT] = 1;
+	    opts[PROMPTSUBST] = opts[PROMPTBANG] = 0;
+
 	    *ep = '\0';
 	    /* expand the contents of the argument so you can use
@@ -14,4 +20,8 @@
 	    free(col);
 	    bv->fm = ep;
+
+	    opts[PROMPTSUBST] = ops;
+	    opts[PROMPTBANG] = opb;
+	    opts[PROMPTPERCENT] = opp;
 	} else {
 	    arg = match_colour((const char **)&bv->fm, is_fg, 0);
```

### not_a_to_b | linux idx 207753 | kind add_only | B ['init'] | outside []

```diff
--- 
+++ 
@@ -34,4 +34,5 @@
 
 	buf->ops = &page_cache_pipe_buf_ops;
+	buf->flags = 0;
 	get_page(page);
 	buf->page = page;
```

### not_a_to_b | linux idx 214336 | kind add_only | B ['release'] | outside []

```diff
--- 
+++ 
@@ -13,7 +13,10 @@
 	}
 
+	mutex_lock(&pfkey_mutex);
 	xfrm_probe_algs();
 
 	supp_skb = compose_sadb_supported(hdr, GFP_KERNEL | __GFP_ZERO);
+	mutex_unlock(&pfkey_mutex);
+
 	if (!supp_skb) {
 		if (hdr->sadb_msg_satype != SADB_SATYPE_UNSPEC)
```

### not_a_to_b | linux idx 210527 | kind del_only | B ['other'] | outside []

```diff
--- 
+++ 
@@ -59,5 +59,4 @@
 	can_free_echo_skb(priv->netdev, ctx->ndx, NULL);
 	mcba_usb_free_ctx(ctx);
-	dev_kfree_skb(skb);
 	stats->tx_dropped++;
 
```

### not_a_to_b | gpac idx 196764 | kind modify | B ['init'] | outside []

```diff
--- 
+++ 
@@ -33,9 +33,10 @@
 			prop_size -= 6;
 			ISOM_DECREASE_SIZE_NO_ERR(ptr, prop_size)
-			//add 2 extra bytes for UTF16 case string dump
-			data2 = gf_malloc(sizeof(char) * (prop_size+2));
+			//add 3 extra bytes for UTF16 case string dump (3 because we need 0-aligned short value)
+			data2 = gf_malloc(sizeof(char) * (prop_size+3));
 			gf_bs_read_data(bs, data2, prop_size);
 			data2[prop_size] = 0;
 			data2[prop_size+1] = 0;
+			data2[prop_size+2] = 0;
 			tag_size-=prop_size;
 		} else {
```

### trivial | ghostpdl idx 210420 | kind trivial | B [] | outside []

```diff
--- 
+++ 
@@ -1,3 +1,3 @@
-fill_threshhold_buffer(byte *dest_strip, byte *src_strip, int src_width,
+fill_threshold_buffer(byte *dest_strip, byte *src_strip, int src_width,
                        int left_offset, int left_width, int num_tiles,
                        int right_width)
```

### trivial | vim idx 206677 | kind trivial | B [] | outside []

```diff
--- 
+++ 
@@ -58,5 +58,5 @@
 			 && (vim_strchr((char_u *)"*?[{~$", *path_end) != NULL
 			     || (!p_fic && (flags & EW_ICASE)
-					     && isalpha(PTR2CHAR(path_end)))))
+					  && vim_isalpha(PTR2CHAR(path_end)))))
 	    e = p;
 	if (has_mbyte)
```

### trivial | xserver idx 210866 | kind trivial | B [] | outside []

```diff
--- 
+++ 
@@ -21,5 +21,5 @@
 
         from.c8 = (CARD8 *) &stuff[1];
-        dataLeft = (stuff->length * 4) - SIZEOF(xkbSelectEventsReq);
+        dataLeft = (client->req_len * 4) - SIZEOF(xkbSelectEventsReq);
         maskLeft = (stuff->affectWhich & (~XkbMapNotifyMask));
         for (ndx = 0, bit = 1; (maskLeft != 0); ndx++, bit <<= 1) {
```

### trivial | linux idx 201343 | kind trivial | B [] | outside []

```diff
--- 
+++ 
@@ -2,5 +2,5 @@
 {
 	return avc_has_perm(&selinux_state,
-			    task_sid_subj(parent), task_sid_obj(current),
+			    task_sid_obj(parent), task_sid_obj(current),
 			    SECCLASS_PROCESS, PROCESS__PTRACE, NULL);
 }
```

### trivial | xserver idx 210834 | kind trivial | B [] | outside []

```diff
--- 
+++ 
@@ -3,4 +3,6 @@
     char *logFileName = NULL;
 
+    /* the format string below is controlled by the user,
+       this code should never be called with elevated privileges */
     if (asprintf(&logFileName, fname, idstring) == -1)
         FatalError("Cannot allocate space for the log file name\n");
```

### trivial | radare2 idx 207780 | kind trivial | B [] | outside []

```diff
--- 
+++ 
@@ -104,5 +104,6 @@
 				continue;
 			}
-			ut64 pa = va2pa (img[j].address, hdr->mappingCount, &cache->maps[maps_index], cache->buf, 0, NULL, NULL);
+			// ut64 pa = va2pa (img[j].address, hdr->mappingCount, &cache->maps[maps_index], cache->buf, 0, NULL, NULL);
+			ut64 pa = va2pa (img[j].address, cache->n_maps, &cache->maps[maps_index], cache->buf, 0, NULL, NULL);
 			if (pa == UT64_MAX) {
 				continue;
```

### trivial | rizin idx 208535 | kind trivial | B [] | outside []

```diff
--- 
+++ 
@@ -1,5 +1,5 @@
-static RzList *relocs(RzBinFile *bf) {
+static RzList *maps(RzBinFile *bf) {
 	rz_return_val_if_fail(bf && bf->o, NULL);
 	QnxObj *qo = bf->o->bin_obj;
-	return rz_list_clone(qo->fixups);
+	return rz_list_clone(qo->maps);
 }
```

### trivial | unicorn idx 210278 | kind trivial | B [] | outside []

```diff
--- 
+++ 
@@ -9,5 +9,5 @@
     //}
 
-    QLIST_REMOVE(block, next);
+    QLIST_REMOVE_RCU(block, next);
     uc->ram_list.mru_block = NULL;
     /* Write list before version */
```
